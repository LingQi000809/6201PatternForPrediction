import os
import random
import shutil
from markov import VariableOrderMarkov
from preprocessing import get_seqs_from_csv, get_onset_offset
from postprocessing import sequence_to_midi, midi_to_wav

# initialize parallel markov models for pitch and onset respectively
max_order = 6
generate_length = 30
pitch_markov = VariableOrderMarkov(max_order=max_order)
onset_markov = VariableOrderMarkov(max_order=max_order)

# get all CSV files from prime and continuation directory
prime_csv_dir = "./datasets/PPDD-Jul2018_aud_mono_small/prime_csv"
cont_csv_dir = "./datasets/PPDD-Jul2018_aud_mono_small/cont_true_csv"
prime_midi_dir = "./datasets/PPDD-Jul2018_aud_mono_small/prime_midi"
cont_midi_dir = "./datasets/PPDD-Jul2018_aud_mono_small/cont_true_midi"
prime_wav_dir = "./datasets/PPDD-Jul2018_aud_mono_small/prime_wav"
cont_wav_dir = "./datasets/PPDD-Jul2018_aud_mono_small/cont_true_wav"
prime_csv_files = [f for f in os.listdir(prime_csv_dir) if f != ".DS_Store"]
cont_csv_files = [f for f in os.listdir(cont_csv_dir) if f != ".DS_Store"]
num_files = len(prime_csv_files)
assert num_files == len(cont_csv_files), "prime directory and continuation directory must have the same number of files"

test_file_index = random.randrange(start=0, stop=num_files)
test_file_id = prime_csv_files[test_file_index]
print(f"Randomly chosen {test_file_id} as the test file.")
assert test_file_id == cont_csv_files[test_file_index], "prime directory and continuation directory must have the same number of files in the same order" 

# train both markov models
for i in range(num_files):
    if prime_csv_files[i] == test_file_id:
        print("skipping the test file during training")
        break
    prime_file = os.path.join(prime_csv_dir, prime_csv_files[i])
    cont_file = os.path.join(cont_csv_dir, cont_csv_files[i])
    onset_offset = get_onset_offset(prime_file)
    prime_pitch_seq, prime_onset_seq = get_seqs_from_csv(prime_file, onset_offset)
    cont_pitch_seq, cont_onset_seq = get_seqs_from_csv(cont_file, onset_offset)
    pitch_markov.train(prime_pitch_seq + cont_pitch_seq)
    onset_markov.train(prime_onset_seq + cont_onset_seq)

# generate for test file
test_prime_csv = os.path.join(prime_csv_dir, test_file_id)
test_cont_csv = os.path.join(cont_csv_dir, test_file_id)
onset_offset = get_onset_offset(test_prime_csv)

pitch_seq_prime, onset_seq_prime = get_seqs_from_csv(test_prime_csv, onset_offset)
true_pitch_cont, true_onset_cont = get_seqs_from_csv(test_cont_csv, onset_offset)
generated_pitch_cont = pitch_markov.generate(generate_length, seq_prime=pitch_seq_prime)
generated_onset_cont = onset_markov.generate(generate_length, seq_prime=onset_seq_prime)

# write outputs
output_dir = f"./test_generation/{test_file_id}"
os.makedirs(output_dir, exist_ok=True)
midi_filepath = sequence_to_midi(
    os.path.join(output_dir, "generated_cont.mid"), 
    onset_offset,
    generated_pitch_cont,
    generated_onset_cont,
    csv_file_path=os.path.join(output_dir, "generated_cont.csv"),
)
wav_filepath = midi_to_wav(
    midi_filepath,
    os.path.join(output_dir, "generated_cont.wav")
)
# copy prime and true continuation files
shutil.copy(test_prime_csv, os.path.join(output_dir, "prime.csv"))
shutil.copy(test_cont_csv, os.path.join(output_dir, "true_cont.csv"))
test_file_id_raw = test_file_id.split(".")[0]
shutil.copy(os.path.join(prime_midi_dir, f"{test_file_id_raw}.mid"), os.path.join(output_dir, "prime.midi"))
shutil.copy(os.path.join(cont_midi_dir, f"{test_file_id_raw}.mid"), os.path.join(output_dir, "true_cont.midi"))
shutil.copy(os.path.join(prime_wav_dir, f"{test_file_id_raw}.wav"), os.path.join(output_dir, "prime.wav"))
shutil.copy(os.path.join(cont_wav_dir, f"{test_file_id_raw}.wav"), os.path.join(output_dir, "true_cont.wav"))
