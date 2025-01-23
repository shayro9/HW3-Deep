r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=200,
        seq_len=64,
        h_dim=256,
        n_layers=3,
        dropout=0.6,
        learn_rate=0.001,
        lr_sched_factor=0.5,
        lr_sched_patience=2
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "Gerzon"
    temperature = 0.0001
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
First, our chorus is very long, processing such text requires storing gradients for all steps during backpropagation,
which uses a lot of memory.
Second, we split the chorus into fixed-length sequences and group them into batches. This way, at every step, 
the model processes a batch of characters from different sequences, allowing training to run in parallel.
Finally, shorter text sequences make the model focus on learning local patterns rather than trying to memorize the 
entire text structure.
"""

part1_q2 = r"""
**Your answer:**
This is because during the text generation we pass the hidden state through each token generation thus making the hidden
state to act kind of as the "memory" of the model and then if we can make the hidden dimension larger than
the sequence size we can increase the memory of the model far beyond the sequence size.
"""

part1_q3 = r"""
**Your answer:**
We want the hidden states to carry context from one timestep to the next. But if batches are shuffled, 
the model receives disjointed sequences, breaking the continuity of the hidden states.
"""

part1_q4 = r"""
**Your answer:**
1. We can see from the graph we plot after implementing the hot soft max that the lower the temperature is, the "sharper"
the graph looks. This means that as we lower the temperature, probability of high-confidence tokens. 
This reduces randomness and making the output aligned with the model’s strongest learned patterns.
2. We can also see from the graph that for a very high temperature, the graph is flatten making the sampling purly random
thus making the model training useless.
3. On the other hand, reducing the temperature to very low values, sharpen the graph to a single highest value (the larger one)
this makes the model to always choose his strongest learned pattern so far thus reducing the exploration of the model 
which can lead to over fitting of the trained data.  

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0, 0)  # (a,b)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**


"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

"""


# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""

PART3_CUSTOM_DATA_URL = None


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim=0,
        num_heads=0,
        num_layers=0,
        hidden_dim=0,
        window_size=0,
        droupout=0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======

    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""

part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""

part4_q3 = r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""

# ==============
