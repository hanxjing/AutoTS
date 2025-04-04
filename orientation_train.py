import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from orientation_net import OrientationNet, TS_localization
from orientation_dataset import TSDataset
from yacs.config import CfgNode as CN

def train(cfg, model, train_loader, val_loader):
    device = cfg.MODEL.DEVICE
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            labels = torch.tensor(batch['direction']).to(device)
            logits = model(batch)

            class_num = np.array(cfg.SOLVER.CLASS_NUM)
            weight = (1.0 / class_num) / np.sum(1.0 / class_num) * 3
            weight = torch.tensor(weight, dtype=torch.float).to(device)
            loss = model.compute_loss(logits, labels, weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Train] Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
        errors = []
        if epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            all_geolocations = []
            all_location_points = []
            all_categories = []

            for batch in train_loader:
                all_geolocations.extend(batch['geolocation'])
                all_location_points.extend(batch['location_point'])
                all_categories.extend(batch['category'])

            geo_data = [
                {
                    'geolocation': geo,
                    'location_point': loc,
                    'category': cat
                }
                for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
            ]

            errors, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)
            print(f"MAE of train: {mae:.2f} m")
            print(f"RMSE        : {rmse:.2f} m")
            print(f"Recall@1m   : {recall_1m * 100:.2f} %")
            print(f"Recall@2m   : {recall_2m * 100:.2f} %")

        evaluate(cfg, model, val_loader, train_loader, epoch, errors)





def evaluate(cfg, model, test_loader, train_loader, epoch=0, errors_train=[], num_classes=3):
    model.eval()
    correct = 0
    total = 0
    correct_per = [0] * num_classes
    total_per = [0] * num_classes
    device = cfg.MODEL.DEVICE

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):

            labels = torch.tensor(batch['direction']).to(device)
            logits = model(batch)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for i in range(num_classes):
                total_per[i] += (labels == i).sum().item()
                correct_per[i] += (preds == labels)[labels == i].sum().item()

    acc = correct / total
    print(f"[Eval] Accuracy: {acc*100:.2f}%")
    if acc > 0.85:
        torch.save(model.state_dict(), 'model_{}.pth'.format(epoch))
        print('ckpt saved')

    mean_accuracy = 0
    for i in range(num_classes):
        accuracy = 100 * correct_per[i] / total_per[i]
        print('Accuracy of class {}: {:.2f}%'.format(i, accuracy))
        mean_accuracy += accuracy
    print('Mean Accuracy of three classes: {:.2f}%'.format(mean_accuracy / num_classes))


    if epoch == cfg.SOLVER.MAX_EPOCHS - 1:
        all_geolocations = []
        all_location_points = []
        all_categories = []

        for batch in test_loader:
            all_geolocations.extend(batch['geolocation'])
            all_location_points.extend(batch['location_point'])
            all_categories.extend(batch['category'])

        geo_data = [
            {
                'geolocation': geo,
                'location_point': loc,
                'category': cat
            }
            for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
        ]

        errors, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)
        print(f"MAE of test: {mae:.2f} m")
        print(f"RMSE       : {rmse:.2f} m")
        print(f"Recall@1m  : {recall_1m * 100:.2f} %")
        print(f"Recall@2m  : {recall_2m * 100:.2f} %")

        if len(errors_train) == 0:
            for batch in train_loader:
                all_geolocations.extend(batch['geolocation'])
                all_location_points.extend(batch['location_point'])
                all_categories.extend(batch['category'])

            geo_data = [
                {
                    'geolocation': geo,
                    'location_point': loc,
                    'category': cat
                }
                for geo, loc, cat in zip(all_geolocations, all_location_points, all_categories)
            ]

            errors_train, mae, rmse, recall_1m, recall_2m = TS_localization(geo_data)

        errors_all = np.concatenate([errors, errors_train])
        mae = np.mean(errors_all)
        rmse = np.sqrt(np.mean(errors_all ** 2))
        recall_1m = np.mean(errors_all < 1.0)
        recall_2m = np.mean(errors_all < 2.0)
        print(f"MAE of all: {mae:.2f} m")
        print(f"RMSE      : {rmse:.2f} m")
        print(f"Recall@1m : {recall_1m * 100:.2f} %")
        print(f"Recall@2m : {recall_2m * 100:.2f} %")


def custom_collate_fn(batch):
    return {
        'image_feature': [item['image_feature'] for item in batch],
        'roi_feature': [item['roi_feature'] for item in batch],
        'sift_feature': [item['sift_feature'] for item in batch],
        'direction': [item['direction'] for item in batch],
        'geolocation': [item['geolocation'] for item in batch],
        'location_point': [item['location_point'] for item in batch],
        'category': [item['category'] for item in batch]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./orientation_config.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    args = parser.parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.cfg)

    device = cfg.MODEL.DEVICE

    # Dataset
    train_set = TSDataset(cfg.DATA.TRAIN_JSON,
                          cfg.DATA.TRAIN_IMG_ROOT,
                          cfg.MODEL.DEVICE,
                          cfg.DETECTRON2.CONFIG_PATH,
                          cfg.DETECTRON2.WEIGHT_PATH,
                          training=True)
    train_loader = DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                  collate_fn=custom_collate_fn)

    val_set = TSDataset(cfg.DATA.TEST_JSON, cfg.DATA.TEST_IMG_ROOT, cfg.MODEL.DEVICE, cfg.DETECTRON2.CONFIG_PATH, cfg.DETECTRON2.WEIGHT_PATH)
    val_loader = DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, collate_fn=custom_collate_fn)

    # Model
    model = OrientationNet(
        roi_input_size=cfg.MODEL.ROI_INPUT_SIZE,
        sift_input_size=cfg.MODEL.SIFT_INPUT_SIZE,
        roi_hidden=cfg.MODEL.ROI_HIDDEN,
        sift_hidden=cfg.MODEL.SIFT_HIDDEN,
        img_hidden=cfg.MODEL.IMG_HIDDEN,
        num_classes=cfg.MODEL.NUM_CLASSES,
        config_path=cfg.DETECTRON2.CONFIG_PATH,
        weight_path=cfg.DETECTRON2.WEIGHT_PATH,
        num_layers=cfg.MODEL.NUM_LAYERS,
        d_model=cfg.MODEL.D_MODEL,
        nhead=cfg.MODEL.NHEAD,
        device=device
    ).to(device)

    if args.eval_only:
        evaluate(cfg, model, val_loader, train_loader)
    else:
        train(cfg, model, train_loader, val_loader)

if __name__ == '__main__':
    main()