import torch.nn as nn



class TransformerModel(nn.Module):
    def __init__(self, roi_input_size, sift_input_size, roi_hidden, sift_hidden, img_hidden, num_classes, num_layers=2, d_model=128, nhead=4):
        super().__init__()
        self.roi_input_size = roi_input_size
        self.sift_input_size = sift_input_size
        self.roi_hidden = roi_hidden
        self.sift_hidden = sift_hidden
        self.img_hidden = img_hidden
        self.num_classes = num_classes

        self.roi_fc = nn.Linear(roi_input_size, self.roi_hidden)
        self.sift_fc = nn.Linear(sift_input_size, self.sift_hidden)
        self.image_fc = nn.Linear(2048, self.img_hidden)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layers)
        self.out_fc = nn.Linear(d_model + self.img_hidden, num_classes)


    def forward(self, roi_input, sift_inputs, img_input):
        # input_cat = torch.cat((roi_input, sift_inputs), 2)
        # input_emb = self.fc_embed(input_cat)
        # output = self.transformer_encoder(input_emb)
        # output = output.mean(dim=1)

        # ioutput = self.image_fc(img_input)
        # output = torch.cat((output, ioutput), dim=1)
        # output = self.out_fc(output)

        roi_emb = self.roi_fc(roi_input)
        sift_emb = self.sift_fc(sift_inputs)
        input_cat = torch.cat((roi_emb, sift_emb), 2)
        output = self.transformer_encoder(input_cat)
        output = output.mean(dim=1)

        img_emb = self.image_fc(img_input)
        output = torch.cat((output, img_emb), dim=1)
        output = self.out_fc(output)

        return output