import numpy as np
from collections import defaultdict

class VariableOrderMarkov:
    """
    Variable-order Markov Model.
    Stores n-grams probability distribution for all orders from 1..max_order.
    Allows generating sequences with lower order than training order.
    """

    def __init__(self, max_order=3):
        # order is the numeric value n in n-gram.
        self.max_order = max_order

        # counts for each n-gram
        #   -> counts[1]: counts for each 1-gram
        #       -> {1-gram context: {next state: count}}
        #   -> counts[2]: counts for each 2-gram
        #       -> {2-gram context: {next state: count}}
        #   -> etc...
        self.counts = {
            k: defaultdict(lambda: defaultdict(float))
            for k in range(1, max_order + 1)
        }

        # transition probability for each n-gram
        self.probs = {
            k: defaultdict(lambda: defaultdict(float))
            for k in range(1, max_order + 1)
        }

        # all unique states (our vocab)
        self.states = set()

        # tracks whether probabilities are up to date.
        self.is_trained = False

    # -----------------------------------------------------
    # Training
    # - called for each audio piece
    # -----------------------------------------------------
    def train(self, seq):
        L = len(seq)
        if L < 2:
            return
        for i in range(L):
            self.states.add(seq[i])
        for order in range(1, self.max_order + 1):
            for i in range(order, L):
                context = tuple(seq[i - order:i])
                nxt = seq[i]
                self.counts[order][context][nxt] += 1
        self.is_trained = False

    # -----------------------------------------------------
    # Probability computation
    # - called after all pieces are trained
    # -----------------------------------------------------
    def compute_probabilities(self):
        for order in range(1, self.max_order + 1):
            for context, next_counts in self.counts[order].items():
                total = sum(next_counts.values())
                if total == 0:
                    continue
                for nxt, count in next_counts.items():
                    self.probs[order][context][nxt] = count / total

        self.is_trained = True

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------
    def generate(self, generate_length, seq_prime=None, order=None):
        """
        generate_length: total number of states to generate after the prime
        order: which order to generate with (1..max_order)
        seq_prime: optional priming sequence (list)
        """
        if not self.is_trained:
            self.compute_probabilities()

        if order is None:
            order = self.max_order

        if seq_prime is None:
            seq_prime = [np.random.choice(list(self.states))]
        prime_len = len(seq_prime)
        seq = list(seq_prime)

        while len(seq) - prime_len < generate_length:
            context = tuple(seq[-order:]) if len(seq) >= order else tuple(seq)

            # Try decreasing orders until we find a valid context
            o = min(order, len(context))
            next_state = None

            while o > 0 and next_state is None:
                c = tuple(context[-o:])
                dist = self.probs[o].get(c, None)
                if dist:
                    next_state = np.random.choice(
                        list(dist.keys()),
                        p=list(dist.values())
                    )
                else:
                    o -= 1

            # If no context found at any order → halt generation
            if next_state is None:
                print(f"Can't find any matching occurrence of context {context}. Halt generation.")
                break

            seq.append(next_state)

        return seq[prime_len:]  # only return the generated continuation
