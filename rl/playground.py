import os
import json
import pprint
import typing
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eyed3
from mutagen.mp3 import MP3
from pydub import AudioSegment
from moviepy.editor import AudioFileClip
from scipy import signal
from scipy.io import wavfile
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from models import *
from retrieval import *


def file_counter(wav_folder_name="fma_small_wav"):
    # folder path
    wav_folder_path = "../data/" + wav_folder_name
    count = 0
    # Iterate directory
    for path in os.listdir(wav_folder_path):
        count += 1
    print('File count:', count)


def mp3_to_wav(mp3_folder_name="fma_small",
               wav_folder_name="fma_small_wav",
               metadata_file_name="metadata.csv"):
    """
    Get Time-Domain representation of music
    i.e. Convert .mp3 data to .wav data

    Parameters
    ----------
    mp3_folder_name: str
        name of the folder that store raw (mp3) data
    wav_folder_name: str
        name of the folder to store converted wav data
    metadata_file_name:str
        name of the file to store metadata of the audio

    Returns
    -------

    """
    mp3_folder_path = "../data/" + mp3_folder_name
    wav_folder_path = "../data/" + wav_folder_name
    metadata_file_path = "../data/" + metadata_file_name

    data = []

    for subset_folder_name in os.listdir(mp3_folder_path):
        subset_folder_path = mp3_folder_path + "/" + subset_folder_name
        print("Process " + subset_folder_name + "...")
        if not os.path.isfile(subset_folder_path):
            # for each mp3 data, get its metadata and convert it to .wav file
            for mp3_file_name in os.listdir(subset_folder_path):
                # read & save metadata
                mp3_file_path = subset_folder_path + "/" + mp3_file_name
                music = MP3(mp3_file_path)
                metadata = {
                    "id": mp3_file_name[:-4],
                    "title": music.get('TIT2').text[0] if music.get('TIT2') else "",
                    "artist": music.get('TPE1').text[0] if music.get('TPE1') else "",
                    "album": music.get('TALB').text[0] if music.get('TALB') else "",
                    "track_number": music.get('TRCK').text[0] if music.get('TRCK') else "",
                    "genre": music.get('TCON').text[0] if music.get('TCON') else "",
                }
                data.append(metadata)

                # convert to .wav
                wav_file_path = wav_folder_path + "/" + mp3_file_name[:-4] + ".wav"
                audio = AudioSegment.from_mp3(mp3_file_path)
                audio.export(wav_file_path, format="wav")

    # save metadata
    df = pd.DataFrame(data)
    print("Number of samples:", len(data))
    df.to_csv(metadata_file_path, index=False)


def wav_to_spectrogram(wav_folder_name="fma_small_wav",
                       spec_folder_name="fma_small_spec"):
    wav_folder_path = "../data/" + wav_folder_name
    spec_folder_path = "../data/" + spec_folder_name
    for wav_file_name in os.listdir(wav_folder_path):
        wav_file_path = wav_folder_path + "/" + wav_file_name
        spec_file_path = spec_folder_path + "/" + wav_file_name[:-4] + ".png"

        # read .wav file
        sample_rate, samples = wavfile.read(wav_file_path)

        # If stereo, select only one channel
        if samples.ndim > 1:
            # samples = samples[:, 0]
            samples = samples.mean(axis=1)

        # calculate the spectrogram
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # display the spectrogram
        plt.figure(figsize=(5.12, 2.56), frameon=False)
        plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading="gouraud", cmap="gray")
        # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
        plt.axis('off')  # turn off axes
        # save the figure without frames as a PNG file
        plt.savefig(spec_file_path, bbox_inches='tight', pad_inches=0)
        # close the figure to free memory
        plt.close()


def data_preprocess(mp3_folder_name="fma_small",
                    wav_folder_name="fma_small_wav",
                    metadata_file_name="metadata.csv"):
    folder_path = "../data/" + mp3_folder_name
    mp3_file_path = folder_path + "/000/000002.mp3"

    # convert mp3 to wav
    # wav_file_path = "../data/" + wav_folder_name + "/000002.wav"
    # sound = AudioSegment.from_mp3(mp3_file_path)
    # sound.export(wav_file_path, format="wav")
    # audio = AudioFileClip(mp3_file_path)
    # audio.write_audiofile(wav_file_path, codec='pcm_s16le')

    # # convert wav to spectrogram
    # sample_rate, samples = wavfile.read(wav_file_path)
    #
    # # If stereo, select only one channel
    # if samples.ndim > 1:
    #     # samples = samples[:, 0]
    #     samples = samples.mean(axis=1)
    #
    # # Calculate the spectrogram
    # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    # # Display the spectrogram
    # plt.figure(figsize=(5.12, 2.56), frameon=False)
    # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading="gouraud", cmap="gray")
    # # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
    # plt.axis("off")
    # plt.savefig("../data/fma_small_spec//000002.png", bbox_inches='tight', pad_inches=0)
    # plt.show(bbox_inches='tight', pad_inches=0)
    # plt.close()


def file_counter_15(wav_folder_name="fma_small_wav"):
    # get audio lengths
    wav_folder_path = "../data/" + wav_folder_name
    audio_lens_15 = []
    audio_lens_30 = []
    audio_less_than_15 = []
    count = 0
    count_15 = 0
    count_30 = 0
    for wav_file_name in os.listdir(wav_folder_path):
        count += 1
        if (count % 1000) == 0:
            print(f"processing {count}-th data ...")
        wav_file_path = wav_folder_path + "/" + wav_file_name
        audio = AudioSegment.from_wav(wav_file_path)
        if len(audio) >= 15000:
            count_15 += 1
            audio_lens_15.append(len(audio))
            if len(audio) >= 30000:
                count_30 += 1
                audio_lens_30.append(len(audio))
        else:
            audio_less_than_15.append(wav_file_path)
    print("-" * 50)
    print("Music shorter than 15 seconds:")
    for audio_ignored in audio_less_than_15:
        print(audio_ignored)
    print("-" * 50)
    print(f"Total number of music: {count}")
    print(f"Number of musics with length >= 15sec: {count_15}")
    print(f"Number of musics with length >= 30sec: {count_30}")
    plt.figure()
    plt.hist(audio_lens_15, bins=100)
    plt.show()


def get_appropriate_data(wav_folder_name="fma_small_wav"):
    """
    Find the appropriate data (audio longer than 15 seconds)

    Parameters
    ----------
    wav_folder_name: str
        name of the folder that store .wav data

    Returns
    -------
    audio_appropriate: list[str]
        path of the appropriate audio
    """
    wav_folder_path = "../data/" + wav_folder_name
    audio_appropriate = []  # store path of the appropriate files
    audio_less_than_15 = []  # store name of the inappropriate files
    count = 0
    count_15 = 0

    for wav_file_name in os.listdir(wav_folder_path):
        count += 1
        if (count % 1000) == 0:
            print(f"checking {count}-th data ...")
        wav_file_path = wav_folder_path + "/" + wav_file_name
        audio = AudioSegment.from_wav(wav_file_path)
        if len(audio) >= 15000:  # audio with length >= 15 sec will be considered as data
            count_15 += 1
            audio_appropriate.append(wav_file_path)
        else:
            audio_less_than_15.append(wav_file_path)

    # print stats
    print("-" * 50)
    print("Music shorter than 15 seconds:")
    for audio in audio_less_than_15:
        print(audio)
    print("-" * 50)
    print(f"Total number of music: {count}")
    print(f"Number of music with length >= 15sec: {count_15}")
    return audio_appropriate


def split_data(file_names, train_ratio=0.8, validation_ratio=0.1, seed=123, output_folder_path="../data/"):
    """
    Splits a list of file names into training, validation, and testing sets.

    Parameters
    ----------
    file_names: list[str]
        The list of file names.
    train_ratio: int
        The ratio of file names to be used for training.
    validation_ratio: int
        The ratio of file names to be used for validation.
    seed: int
        Random seed for reproducibility
    output_folder_path: str
        The output folder path

    Returns
    -------
    data_splits: dict[str, list[str]]
        A dictionary containing the three splits.
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Shuffle the list to ensure randomness
    random.shuffle(file_names)

    # Calculate the number of files for each set
    total_files = len(file_names)
    train_size = int(total_files * train_ratio)
    validation_size = int(total_files * validation_ratio)
    test_size = total_files - train_size - validation_size

    # Split the list
    train_files = file_names[:train_size]
    validation_files = file_names[train_size:train_size + validation_size]
    test_files = file_names[train_size + validation_size:]

    # save the split result
    data_splits = {
        "train": train_files,
        "valid": validation_files,
        "test": test_files
    }
    if output_folder_path:
        output_file_path = output_folder_path + "data_splits.json"
        with open(output_file_path, 'w') as f:
            json.dump(data_splits, f)

    return data_splits


def extract_random_segment(data_splits_path="../data/data_splits.json",
                           segment_length=15000,
                           wav_folder_name="fma_small_wav",
                           wav_segment_folder_name="fma_small_wav_rand15"):
    """
    Randomly extract 15-seconds clip from each of the data

    Parameters
    ----------
    data_splits_path: str
        path to the file that store data_splits
        note: data_splits is a dictionary that store how data are split
    segment_length: int
        length of the clip to be extracted
    wav_folder_name: str
        name of the folder that store .wav data
    wav_segment_folder_name: str
        name of the folder to store extracted 15-sec clip

    Returns
    -------

    """
    wav_folder_path = "../data/" + wav_folder_name + "/"
    wav_file_name_start = len(wav_folder_path)  # start index of the name of wav file
    wav_segment_train_folder_path = "../data/" + wav_segment_folder_name + "/train/"
    wav_segment_valid_folder_path = "../data/" + wav_segment_folder_name + "/valid/"
    wav_segment_test_folder_path = "../data/" + wav_segment_folder_name + "/test/"

    # get data splits
    with open(data_splits_path, 'r') as file:
        data_splits = json.load(file)

    # Extract 15-sec segments:
    print("-" * 50)
    print("Extracting segments for training set ...")
    for wav_file_path in data_splits["train"]:
        # Load audio
        audio = AudioSegment.from_wav(wav_file_path)

        # Choose a random start point for the 15-second segment
        start_point = random.randint(0, len(audio) - segment_length)
        segment = audio[start_point:start_point + segment_length]

        # Save the 15-second segment
        wav_segment_file_path = wav_segment_train_folder_path + wav_file_path[wav_file_name_start:]
        segment.export(wav_segment_file_path, format="wav")
    print("Done.")

    print("-" * 50)
    print("Extracting segments for validation set ...")
    for wav_file_path in data_splits["valid"]:
        # Load audio
        audio = AudioSegment.from_wav(wav_file_path)

        # Choose a random start point for the 15-second segment
        start_point = random.randint(0, len(audio) - segment_length)
        segment = audio[start_point:start_point + segment_length]

        # Save the 15-second segment
        wav_segment_file_path = wav_segment_valid_folder_path + wav_file_path[wav_file_name_start:]
        segment.export(wav_segment_file_path, format="wav")
    print("Done.")

    print("-" * 50)
    print("Extracting segments for test set ...")
    for wav_file_path in data_splits["test"]:
        # Load audio
        audio = AudioSegment.from_wav(wav_file_path)

        # Choose a random start point for the 15-second segment
        start_point = random.randint(0, len(audio) - segment_length)
        segment = audio[start_point:start_point + segment_length]

        # Save the 15-second segment
        wav_segment_file_path = wav_segment_test_folder_path + wav_file_path[wav_file_name_start:]
        segment.export(wav_segment_file_path, format="wav")
    print("Done.")


def wav_segment_to_spectrogram(wav_segment_folder_name="fma_small_wav_rand15",
                               spec_folder_name="fma_small_spec_rand15"):
    """
    Convert the 15-seconds audio clip (.wav) to spectrogram

    Parameters
    ----------
    wav_segment_folder_name: str
        name of the folder that store extracted 15-sec clip
    spec_folder_name: str
        name of the folder to store converted spectrogram

    Returns
    -------

    """
    wav_segment_train_folder_path = "../data/" + wav_segment_folder_name + "/train/"
    wav_segment_valid_folder_path = "../data/" + wav_segment_folder_name + "/valid/"
    wav_segment_test_folder_path = "../data/" + wav_segment_folder_name + "/test/"
    wav_segment_folder_paths = [wav_segment_train_folder_path,
                                wav_segment_valid_folder_path,
                                wav_segment_test_folder_path]

    spec_train_folder_path = "../data/" + spec_folder_name + "/train/"
    spec_valid_folder_path = "../data/" + spec_folder_name + "/valid/"
    spec_test_folder_path = "../data/" + spec_folder_name + "/test/"
    spec_folder_paths = [spec_train_folder_path, spec_valid_folder_path, spec_test_folder_path]

    dataset_names = ["train", "valid", "test"]

    for i in range(len(wav_segment_folder_paths)):
        print("-" * 50)
        print(f"Transforming {dataset_names[i]} data to spectrogram ...")
        count = 0
        wav_segment_folder_path = wav_segment_folder_paths[i]
        spec_folder_path = spec_folder_paths[i]
        for wav_file_name in os.listdir(wav_segment_folder_path):
            count += 1
            if (count % 1000) == 0:
                print(f"transforming {count}-th data ...")
            wav_file_path = wav_segment_folder_path + wav_file_name
            spec_file_path = spec_folder_path + wav_file_name[:-4] + ".png"

            # read .wav file
            sample_rate, samples = wavfile.read(wav_file_path)

            # If stereo, select only one channel
            if samples.ndim > 1:
                # samples = samples[:, 0]
                samples = samples.mean(axis=1)

            # calculate the spectrogram
            frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
            # display the spectrogram
            plt.figure(figsize=(5.12, 2.56), frameon=False)
            # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading="gouraud", cmap="gray")
            plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
            plt.axis('off')  # turn off axes
            # Set frequency limits
            plt.ylim(0, 12000)
            # save the figure without padding as a PNG file
            plt.tight_layout(pad=0)
            plt.savefig(spec_file_path)
            # close the figure to free memory
            plt.close()
        print("Done.")


def extract_mid_segment(data_splits_path="../data/data_splits.json",
                        segment_length=15000,
                        wav_folder_name="fma_small_wav",
                        wav_segment_folder_name="fma_small_wav_mid15"):
    wav_folder_path = "../data/" + wav_folder_name + "/"
    wav_file_name_start = len(wav_folder_path)  # start index of the name of wav file
    wav_segment_folder_path = "../data/" + wav_segment_folder_name + "/"

    # create folder if not exists
    if not os.path.exists(wav_segment_folder_path):
        os.makedirs(wav_segment_folder_path)

    # get data splits
    with open(data_splits_path, 'r') as file:
        data_splits = json.load(file)

    print("-" * 50)
    print("Extracting mid segments for test set ...")
    for wav_file_path in data_splits["test"]:
        # Load the audio file
        audio = AudioSegment.from_wav(wav_file_path)
        audio_length = len(audio)
        audio_id = wav_file_path[wav_file_name_start:].split('.')[0]

        if 15000 <= audio_length < 30000:
            # If the audio is between 15 and 30 seconds, extract the middle 15 seconds
            start = (audio_length - segment_length) // 2
            end = start + segment_length
            mid_segment = audio[start:end]

            wav_segment_file_path = wav_segment_folder_path + audio_id + "_1.wav"
            mid_segment.export(wav_segment_file_path, format="wav")
        elif audio_length >= 30000:
            # If the audio is 30 seconds or longer, extract two middle 15-second clips
            middle = audio_length // 2
            first_segment_start = middle - segment_length
            first_segment_end = middle
            second_segment_start = middle
            second_segment_end = middle + segment_length

            mid_segment1 = audio[first_segment_start:first_segment_end]
            mid_segment2 = audio[second_segment_start:second_segment_end]

            wav_segment_file_path1 = wav_segment_folder_path + audio_id + "_1.wav"
            wav_segment_file_path2 = wav_segment_folder_path + audio_id + "_2.wav"
            mid_segment1.export(wav_segment_file_path1, format="wav")
            mid_segment2.export(wav_segment_file_path2, format="wav")
    print("Done.")


def wav_mid_segment_to_spectrogram(wav_segment_folder_name="fma_small_wav_mid15",
                                   spec_folder_name="fma_small_spec_mid15"):
    wav_segment_folder_path = "../data/" + wav_segment_folder_name + "/"
    spec_folder_path = "../data/" + spec_folder_name + "/"

    # create folder if not exists
    if not os.path.exists(spec_folder_path):
        os.makedirs(spec_folder_path)

    print("-" * 50)
    print("Transforming test data (mid segments) to spectrogram ...")
    for wav_file_name in os.listdir(wav_segment_folder_path):
        wav_file_path = wav_segment_folder_path + wav_file_name
        spec_file_path = spec_folder_path + wav_file_name[:-4] + ".png"

        # read .wav file
        sample_rate, samples = wavfile.read(wav_file_path)

        # If stereo, select only one channel
        if samples.ndim > 1:
            # samples = samples[:, 0]
            samples = samples.mean(axis=1)

        # calculate the spectrogram
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # display the spectrogram
        plt.figure(figsize=(5.12, 2.56), frameon=False)
        # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading="gouraud", cmap="gray")
        plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
        plt.axis('off')  # turn off axes
        # Set frequency limits
        plt.ylim(0, 12000)
        # save the figure without padding as a PNG file
        plt.tight_layout(pad=0)
        plt.savefig(spec_file_path)
        # close the figure to free memory
        plt.close()
    print("Done.")


def test():
    # directory = "../data/fma_small_wav_rand15/train/"
    # for f in os.listdir(directory):
    #     print(f)
    #     audio_path = os.path.join(directory, f)
    #     print(audio_path)
    #     # audio = AudioSegment.from_wav(audio_path)
    #     sample_rate, samples = wavfile.read(audio_path)
    #     samples = samples.mean(axis=1)
    #     print(samples.shape)
    #
    #     to_tensor = transforms.ToTensor()
    #
    #     print(os.path.splitext(f)[0])
    #     break

    autoencoder = WavAutoEncoder()
    input_tensor = torch.randn(1, 1, 661500)  # Example input
    output = autoencoder(input_tensor)
    print(output.shape)  # Should be torch.Size([1, 1, 661500])
    rep = autoencoder.encode(input_tensor)
    print(rep.shape)


    # # Use GPU if possible
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print('There are %d GPU(s) available.' % torch.cuda.device_count())
    #     print('Using GPU:', torch.cuda.get_device_name(0))
    # else:
    #     device = torch.device("cpu")
    #     print('No GPU available, using the CPU instead.')
    #
    # model = SpecAutoEncoder()
    # model_name = "SpecAutoEncoder"
    # model.load_state_dict(torch.load("../models" + f"/{model_name}_best.pt"))
    # model.to(device)
    #
    # retrieval_model = RetrievalModel(model, device, database_path="../data/databases/spec/")
    # test_dataset = torch.load("../data/datasets/spec/test.pt")
    # test_dataloader = DataLoader(test_dataset, batch_size=1)
    # recall1, recall5, recall20 = retrieval_model.evaluate(test_dataloader, num_retrieved=20)
    # print(f"Recall@1: {recall1}")
    # print(f"Recall@5: {recall5}")
    # print(f"Recall@20: {recall20}")

    # ----------------------------------------------------------------------------

    # # 15-sec mid segments wav data to spectrogram
    # wav_mid_segment_to_spectrogram()

    # filename = "062531_1.wav"
    # print(filename.split('.')[0])
    # print(filename.split('_')[0])

    # # extract mid segments
    # extract_mid_segment()

    # # Example of using the autoencoder
    # autoencoder = SpecAutoEncoder()
    # input_tensor = torch.randn(1, 3, 256, 512)  # Example input
    # output = autoencoder(input_tensor)
    # print(output.shape)  # Should be torch.Size([1, 3, 256, 512])

    # directory = "../data/fma_small_spec_rand15/train/"
    # for f in os.listdir(directory):
    #     print(f)
    #     img_name = os.path.join(directory, f)
    #     print(img_name)
    #     image = Image.open(img_name).convert('RGB')
    #     print(image.size)
    #
    #     to_tensor = transforms.ToTensor()
    #     image = to_tensor(image)
    #     print(image.shape)
    #     print(torch.max(image))
    #     print(torch.min(image))
    #
    #     image = image.cpu().clone()  # Clone the tensor to not modify the original
    #     # image = image.squeeze(0)  # Remove the batch dimension
    #     image = to_pil_image(image)  # Convert to PIL image
    #
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.show()
    #
    #     print(os.path.splitext(f)[0])
    #     break

    # # 15-sec segments wav data to spectrogram
    # wav_segment_to_spectrogram()

    # # get 15-sec segments wav data
    # extract_random_segment()

    # # find appropriate data and split it
    # random_seed = 123
    # audio_appropriate = get_appropriate_data()
    # data_splits = split_data(audio_appropriate, seed=random_seed)
    # print(len(data_splits["train"]))
    # print(len(data_splits["valid"]))
    # print(len(data_splits["test"]))

    # file_counter_15()
    # files_to_delete = ["file_to_delete.txt", "file_to_delete2.txt"]
    # for f in files_to_delete:
    #     os.remove(f)

    # data_preprocess()
    # mp3_to_wav()
    # wav_to_spectrogram()


if __name__ == '__main__':
    test()
