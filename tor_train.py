# train_minimal_lpr.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model.LPRNet import build_lprnet


# -------- Config ----------
BATCH_SIZE = 32
NUM_BATCHES = 30            # number of batches in one epoch (fake dataset)
IMG_SHAPE = (3, 24, 94)     # C,H,W
CLASS_NUM = 37              # number of classes (including blank at last index)
MAX_LABEL_LEN = 10           # max characters per sample (fake)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
# --------------------------


def get_model(class_num):
    # adapt to builder signature if present in your repo (args might differ)
    return build_lprnet(lpr_max_len=18, phase='train', class_num=class_num, dropout_rate=0.5)


# --------- Fake dataset (realistic structure) ----------
class FakeLPRDataset(Dataset):
    """
    Returns:
      - image: FloatTensor (3,24,94)
      - label: 1D LongTensor (variable length)
      - length: int (length of that label)
    """
    def __init__(self, num_samples, img_shape=IMG_SHAPE, max_len=MAX_LABEL_LEN, class_num=CLASS_NUM):
        self.num_samples = num_samples
        self.img_shape = img_shape
        self.max_len = max_len
        self.class_num = class_num
        np.random.seed(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # random image like real input (N,3,24,94)
        img = np.random.randn(*self.img_shape).astype(np.float32)
        # random label length between 1 and max_len
        L = np.random.randint(1, self.max_len + 1)
        # labels are integers in [0, class_num-2]; last index (class_num-1) reserved for blank
        labels = np.random.randint(0, self.class_num - 1, size=(L,), dtype=np.int64)
        return torch.from_numpy(img), torch.from_numpy(labels), L

def collate_fn(batch):
    """
    Convert a list of samples into the inputs required by CTCLoss:
    - images : Tensor (N, C, H, W)
    - labels_concatenated : 1D LongTensor (sum(target_lengths))
    - target_lengths : list or LongTensor of lengths
    """
    imgs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    lengths = [b[2] for b in batch]
    imgs = torch.stack(imgs, dim=0)           # (N,C,H,W)
    # concatenate labels to 1D tensor as expected by PyTorch CTCLoss
    labels_concat = torch.cat(labels).long()
    return imgs, labels_concat, torch.tensor(lengths, dtype=torch.long)

# --------- Utility to create input_lengths for CTC ----------
def make_input_lengths_from_logits(logits):
    # logits: (N, class_num, T)
    T = logits.size(2)
    batch_size = logits.size(0)
    # CTC expects input_lengths per sample (length of T for each sample)
    return torch.full((batch_size,), T, dtype=torch.long)

# --------- Minimal training loop (1 epoch) ----------
def train_one_epoch():
    model = get_model(CLASS_NUM)
    model.to(DEVICE)
    model.train()

    optimizer = optim.RMSprop(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-5)
    ctc_loss = nn.CTCLoss(blank=CLASS_NUM-1, reduction='mean')  # blank index is last

    dataset = FakeLPRDataset(num_samples=BATCH_SIZE * NUM_BATCHES, img_shape=IMG_SHAPE, max_len=MAX_LABEL_LEN, class_num=CLASS_NUM)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    running_loss = 0.0
    for batch_idx, (images, labels, target_lengths) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        # forward
        logits = model(images)                       # (N, C, T)
        # prepare log_probs for CTCLoss => (T, N, C)
        log_probs = logits.permute(2, 0, 1).log_softmax(2).requires_grad_()
        input_lengths = make_input_lengths_from_logits(logits).to(DEVICE)

        # compute loss
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 5 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    print(f"Finished 1 epoch, avg loss = {avg_loss:.4f}")

if __name__ == "__main__":
    train_one_epoch()
    print("Done.")
