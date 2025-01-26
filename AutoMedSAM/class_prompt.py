import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dense_attention = nn.Conv2d(128 + 1, 128, kernel_size=1)

        self.dense_dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dense_dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.sparse_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128 + 1, 128, kernel_size=1),
            nn.Sigmoid()
        )

        self.sparse_dec1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.sparse_dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, label):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)

        middle_out = self.middle(enc2_out)

        label_proj_dense = label.view(label.size(0), 1, 1, 1).expand(-1, -1, middle_out.size(2), middle_out.size(3))
        middle_out_dense = torch.cat([middle_out, label_proj_dense], dim=1)
        attention_dense = self.dense_attention(middle_out_dense)
        middle_out_dense = middle_out * attention_dense

        dense_dec1_in = torch.cat([middle_out_dense, enc2_out], dim=1)
        dense_dec1_out = self.dense_dec1(dense_dec1_in)

        dense_dec2_in = torch.cat([dense_dec1_out, enc1_out], dim=1)
        dense_embeddings = self.dense_dec2(dense_dec2_in)

        label_proj_sparse = label.view(label.size(0), 1, 1, 1).expand(-1, -1, middle_out.size(2), middle_out.size(3))
        middle_out_sparse = torch.cat([middle_out, label_proj_sparse], dim=1)
        attention_sparse = self.sparse_attention(middle_out_sparse)
        middle_out_sparse = middle_out * attention_sparse

        sparse_dec1_in = torch.cat([middle_out_sparse, enc2_out], dim=1)
        sparse_dec1_out = self.sparse_dec1(sparse_dec1_in)

        sparse_dec2_in = torch.cat([sparse_dec1_out, enc1_out], dim=1)
        sparse_embeddings = self.sparse_dec2(sparse_dec2_in)

        return dense_embeddings, sparse_embeddings


class DiffusionModel(nn.Module):
    def __init__(self, embedding_channels, timesteps=100):
        super(DiffusionModel, self).__init__()
        self.embedding_channels = embedding_channels
        self.timesteps = timesteps

        self.unet = UNet(in_channels=embedding_channels + 1, out_channels=embedding_channels)

        self.fc_label = nn.Linear(1, 64 * 64)
        self.sparse_out = nn.AdaptiveAvgPool2d((2,1))

    def forward_diffusion(self, x, t, label):
        label_proj = self.fc_label(label.float().view(-1, 1))  # Shape: (B, 1)
        label_proj = label_proj.view(label_proj.size(0), 1, x.size(2), x.size(3))  # Shape: (B, 1, 64, 64)
        label_proj = label_proj.expand(-1, -1, x.size(2), x.size(3))
        noise = torch.randn_like(x) * (1 / (t + 1)) + label_proj
        return x + noise

    def reverse_diffusion(self, noisy_embedding, label):
        label_proj = self.fc_label(label.float().view(-1, 1)).unsqueeze(2).unsqueeze(3)  # Shape: (B, C, 1, 1)
        label_proj = label_proj.view(label_proj.size(0), 1, noisy_embedding.size(2), noisy_embedding.size(3))
        input_with_label = torch.cat([noisy_embedding, label_proj], dim=1)

        dense_embeddings, sparse_embeddings = self.unet(input_with_label, label)
        sparse_embeddings = (self.sparse_out(sparse_embeddings)).squeeze().permute(0, 2, 1)
        return dense_embeddings, sparse_embeddings

    def forward(self, embedding, label):
        B, C, H, W = embedding.size()
        x = embedding

        # Forward diffusion: add noise progressively
        for t in range(self.timesteps, 0, -1):
            x = self.forward_diffusion(x, t, label)

        dense_embeddings, sparse_embeddings = self.reverse_diffusion(x, label)

        return dense_embeddings, sparse_embeddings
if __name__ == '__main__':

    image_encoder_output = torch.randn((4, 256, 64, 64))  # Example image embedding (B, C, H, W)
    labels = torch.randint(low=0, high=10, size=(4, 1))
    medsam_diffusion_model = DiffusionModel(embedding_channels=256, timesteps=7)

    dense_embeddings, sparse_embeddings = medsam_diffusion_model(image_encoder_output, labels)  # Output: (B, 256, 64, 64)
