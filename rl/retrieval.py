import os
import io
import heapq
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


class RetrievalModel:
    def __init__(self, model: nn.Module, device, database_path: str):
        self.model = model
        self.model.eval()
        self.device = device
        self.database_path = database_path

    def retrieve(self, x, num_retrieved=1):
        # Normalize x for cosine similarity calculation
        x_flat = x.flatten()
        x_norm = x_flat / torch.linalg.norm(x_flat)

        # Min heap to maintain top k elements
        min_heap = []

        # Iterate through each file in the folder
        for file_name in os.listdir(self.database_path):
            # Load the representation
            file_path = os.path.join(self.database_path, file_name)
            rep = torch.load(file_path)
            rep = rep.to(self.device)

            # Normalize the loaded representation
            rep_flat = rep.flatten()
            rep_norm = rep_flat / torch.linalg.norm(rep_flat)

            # Compute cosine similarity
            # similarity = cosine_similarity(x_norm, rep_norm)
            similarity = cosine_similarity(x_norm.unsqueeze(0), rep_norm.unsqueeze(0)).item()

            # Parse the id from the filename
            clip_id = file_name.split('_')[0]

            # Add to heap and maintain heap size num_retrieved
            if len(min_heap) < num_retrieved:
                heapq.heappush(min_heap, (similarity, clip_id))
            else:
                heapq.heappushpop(min_heap, (similarity, clip_id))

        # Convert min heap to a sorted list in descending order of similarity
        top_k = sorted(min_heap, key=lambda r: r[0], reverse=True)
        top_k = [item[1] for item in top_k]

        return top_k

    def evaluate(self, test_dataloader, num_retrieved):
        recall1_scores = []
        recall5_scores = []
        recall20_scores = []

        for batch, (x, x_id) in enumerate(test_dataloader):
            if (batch % 100) == 0:
                print(f"Evaluating {batch+1}-th data ...")

            with torch.no_grad():
                x = x.to(self.device)
                x_rep = self.model.encode(x)
            x_id = x_id[0]

            top_k = self.retrieve(x_rep, num_retrieved=num_retrieved)
            recall1 = 1 if x_id in top_k[:1] else 0
            recall5 = 1 if x_id in top_k[:5] else 0
            recall20 = 1 if x_id in top_k[:20] else 0

            recall1_scores.append(recall1)
            recall5_scores.append(recall5)
            recall20_scores.append(recall20)
        print("Done.")

        avg_recall1 = sum(recall1_scores) / len(recall1_scores)
        avg_recall5 = sum(recall5_scores) / len(recall5_scores)
        avg_recall20 = sum(recall20_scores) / len(recall20_scores)
        return avg_recall1, avg_recall5, avg_recall20


def create_spec_database(model: nn.Module, device, input_path: str, database_path: str):
    # create folder if not exists
    if not os.path.exists(database_path):
        os.makedirs(database_path)

    to_tensor = transforms.ToTensor()
    print("-"*50)
    print("Getting representations for data ...")
    for img_name in os.listdir(input_path):
        img_path = input_path + img_name
        img_id = img_name.split('.')[0]

        image = Image.open(img_path).convert('RGB')

        # convert image to tensor
        image = to_tensor(image)
        image = image.unsqueeze(dim=0)

        # get representation
        with torch.no_grad():
            image = image.to(device)
            rep = model.encode(image)
        rep_path = database_path + f"{img_id}.pt"
        torch.save(rep, rep_path)
    print("Done.")


def wav_to_tensor(wav_file_path):
    # Read .wav file
    sample_rate, samples = wavfile.read(wav_file_path)

    # If stereo, take the mean to convert to mono
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    # Calculate the spectrogram
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Create the spectrogram image in memory (without saving to disk)
    plt.figure(figsize=(5.12, 2.56), frameon=False)
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
    plt.axis('off')  # turn off axes
    plt.ylim(0, 12000)
    plt.tight_layout(pad=0)

    # Save the spectrogram image to a buffer instead of a file
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Open the image and convert to 'RGB'
    image = Image.open(buf).convert('RGB')

    # Convert image to tensor
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image)

    # Close the buffer and clear the plot
    buf.close()
    plt.close()

    return tensor


def wav_to_rep(wav_file_path, model, device):
    model.eval()
    x = wav_to_tensor(wav_file_path)
    with torch.no_grad():
        x = x.to(device)
        rep = model.encode(x.unsqueeze(0))
    return rep


def get_database(database_path):
    count = 0
    for file_name in os.listdir(database_path):
        count += 1
    database = torch.zeros(count, 64, 32, 64, dtype=torch.float32)
    ids = []
    for idx, file_name in enumerate(os.listdir(database_path)):
        # Load the representation
        file_path = os.path.join(database_path, file_name)
        rep = torch.load(file_path)
        database[idx] = rep.squeeze()
        ids.append(file_name.split("_")[0])
    torch.save(database, "../data/database.pt")
    with open("../data/tensor_ids.json", 'w') as f:
        json.dump(ids, f)










