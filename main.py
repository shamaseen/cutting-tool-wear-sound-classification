from utils import *
import os
import argparse

# load audio transform
audio_process=Preprocessing()

def main():
    parser = argparse.ArgumentParser(description="Process audio files using a specified model.")
    parser.add_argument("--model_path", type=str, help="Path to the model file")
    parser.add_argument("--audio_path", type=str, help="Path to the audio folder")
    args = parser.parse_args()
    # check is model file and audio folder exist
    print(args.model_path)
    assert os.path.isfile(args.model_path)
    assert os.path.isdir(args.audio_path)
    # load model
    if 'onnx' in args.model_path:
        model=soundModel(args.model_path)
    elif 'pt' in args.model_path:
        model=model_inferance(args.model_path)
    # load audio
    audio_files_paths=[os.path.join(args.audio_path,file) for file in os.listdir(args.audio_path)]
    output=audio_process.generate_sample(audio_files_paths)
    results=model.predict(output)
    print(results)
if __name__ == "__main__":
    main()