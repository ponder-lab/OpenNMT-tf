"""Train a small Transformer with the custom training loop of custom_transformer_training.py,
passing each batch to the training step as an argument.

This is a runnable variant of custom_transformer_training.py, which is left unchanged. It differs
in exactly these ways:

* The training step takes the batch as arguments, `training_step(source, target)`, and the loop
  iterates the dataset in Python, `for source, target in dataset`. In the original, the step takes
  no arguments and advances an iterator inside the tf.function. The step is decorated with a bare
  `@tf.function`: no input_signature is given.
* The model is small (2 layers of 64 units) and training stops after TRAIN_STEPS steps, so a run
  takes about a minute on a CPU.
* It needs no data files: it writes a synthetic parallel corpus of random sentences of varying
  length to a temporary directory, and builds the vocabularies with the package's own
  `opennmt.bin.build_vocab`.

The dataset is built exactly as in the original: token-based batching with length_bucket_width=1,
so batches of different sentence lengths reach the training step with different shapes.
"""

import logging
import os
import random
import subprocess
import sys
import tempfile

import tensorflow as tf
import tensorflow_addons as tfa
import opennmt as onmt

tf.get_logger().setLevel(logging.INFO)

TRAIN_STEPS = 60
NUM_UNITS = 64
NUM_SENTENCES = 600
MAXIMUM_LENGTH = 40


def write_corpus(directory, seed=0):
  """Writes a synthetic parallel corpus and its vocabularies, returning their paths."""
  rng = random.Random(seed)
  words = ["w%d" % i for i in range(200)]
  paths = {name: os.path.join(directory, name) for name in ("src.txt", "tgt.txt")}
  with open(paths["src.txt"], "w") as src, open(paths["tgt.txt"], "w") as tgt:
    for _ in range(NUM_SENTENCES):
      length = rng.randint(3, MAXIMUM_LENGTH)
      sentence = [rng.choice(words) for _ in range(length)]
      src.write(" ".join(sentence) + "\n")
      tgt.write(" ".join(reversed(sentence)) + "\n")
  for name in ("src", "tgt"):
    vocab = os.path.join(directory, name + ".vocab")
    subprocess.run([sys.executable, "-m", "opennmt.bin.build_vocab",
                    "--save_vocab", vocab, paths[name + ".txt"]], check=True)
    paths[name + ".vocab"] = vocab
  return paths


model = onmt.models.SequenceToSequence(
    source_inputter=onmt.inputters.WordEmbedder(embedding_size=NUM_UNITS),
    target_inputter=onmt.inputters.WordEmbedder(embedding_size=NUM_UNITS),
    encoder=onmt.encoders.SelfAttentionEncoder(
        num_layers=2,
        num_units=NUM_UNITS,
        num_heads=4,
        ffn_inner_dim=2 * NUM_UNITS,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1),
    decoder=onmt.decoders.SelfAttentionDecoder(
        num_layers=2,
        num_units=NUM_UNITS,
        num_heads=4,
        ffn_inner_dim=2 * NUM_UNITS,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1))

learning_rate = onmt.schedules.NoamDecay(scale=2.0, model_dim=NUM_UNITS, warmup_steps=8000)
optimizer = tfa.optimizers.LazyAdam(learning_rate)


def train(source_file, target_file):
  dataset = model.examples_inputter.make_training_dataset(
      source_file,
      target_file,
      batch_size=3072,
      batch_type="tokens",
      shuffle_buffer_size=-1,
      length_bucket_width=1,
      maximum_features_length=MAXIMUM_LENGTH,
      maximum_labels_length=MAXIMUM_LENGTH)

  @tf.function
  def training_step(source, target):
    # Run the encoder.
    source_inputs = model.features_inputter(source, training=True)
    encoder_outputs, _, _ = model.encoder(
        source_inputs,
        source["length"],
        training=True)

    # Run the decoder.
    target_inputs = model.labels_inputter(target, training=True)
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=source["length"])
    logits, _, _ = model.decoder(
        target_inputs,
        target["length"],
        state=decoder_state,
        training=True)

    # Compute the cross entropy loss.
    loss_num, loss_den, _ = onmt.utils.cross_entropy_sequence_loss(
        logits,
        target["ids_out"],
        target["length"],
        label_smoothing=0.1,
        average_in_time=True,
        training=True)
    loss = loss_num / loss_den

    # Compute and apply the gradients.
    variables = model.trainable_variables
    gradients = optimizer.get_gradients(loss, variables)
    optimizer.apply_gradients(list(zip(gradients, variables)))
    return loss

  shapes = set()
  for source, target in dataset:
    shapes.add(tuple(source["ids"].shape))
    loss = training_step(source, target)
    step = optimizer.iterations.numpy()
    if step % 10 == 0:
      tf.get_logger().info("Step = %d ; Loss = %f", step, loss)
    if step >= TRAIN_STEPS:
      break
  tf.get_logger().info("Distinct source batch shapes: %d ; training_step traces: %d",
                       len(shapes), training_step.experimental_get_tracing_count())


def main():
  with tempfile.TemporaryDirectory() as directory:
    paths = write_corpus(directory)
    model.initialize({"source_vocabulary": paths["src.vocab"],
                      "target_vocabulary": paths["tgt.vocab"]})
    train(paths["src.txt"], paths["tgt.txt"])


if __name__ == "__main__":
  main()
