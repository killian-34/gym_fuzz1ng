import numpy as np
from collections import deque
from numba import jit

class Queue():

    def __init__(self, inputs):
        self.queue = deque(inputs)

    def enqueue(self, a):
        self.queue.append(a)

    def pop(self):
        return self.queue.popleft()


# bytearray, byte index, bit index
# @jit
# def flip_bit(a,byte_i,bit_i):
#     a[byte_i] ^= (1 << bit_i)

# @jit
# def flip_bits(a, num_bits, byte_i, bit_i):
#     a[byte_i] ^= ((2**num_bits - 1) << bit_i)

@jit
def flip_bit_afl(a, bit_i):
    a[bit_i >> 3] ^= (128 >> (bit_i & 7))

@jit
def flip_byte_afl(a, byte_i):
    a[byte_i] ^= 0xFF

# must take bytearray
# @jit
# jit not working with byte arrays for some reason
def deterministic_edits(orig, out_buff):
    
    yield out_buff

    input_len = len(orig)

    # flip single bits
    for bit_i in range(input_len << 3):
        flip_bit_afl(out_buff, bit_i)
        yield out_buff
        flip_bit_afl(out_buff, bit_i)
    

    # flip two bits
    for bit_i in range(input_len << 3 - 1):
        flip_bit_afl(out_buff, bit_i)
        flip_bit_afl(out_buff, bit_i+1)
        yield out_buff
        flip_bit_afl(out_buff, bit_i)
        flip_bit_afl(out_buff, bit_i+1)

    # flip four bits
    for bit_i in range(input_len << 3 - 3):
        flip_bit_afl(out_buff, bit_i)
        flip_bit_afl(out_buff, bit_i+1)
        flip_bit_afl(out_buff, bit_i+2)
        flip_bit_afl(out_buff, bit_i+3)
        yield out_buff
        flip_bit_afl(out_buff, bit_i)
        flip_bit_afl(out_buff, bit_i+1)
        flip_bit_afl(out_buff, bit_i+2)
        flip_bit_afl(out_buff, bit_i+3)


    # flip byte
    for byte_i in range(input_len):
        flip_byte_afl(out_buff, byte_i)
        yield out_buff
        flip_byte_afl(out_buff, byte_i)

    # flip 2 bytes
    for byte_i in range(input_len-1):
        flip_byte_afl(out_buff, byte_i)
        flip_byte_afl(out_buff, byte_i+1)
        yield out_buff
        flip_byte_afl(out_buff, byte_i)
        flip_byte_afl(out_buff, byte_i+1)

    # flip 4 bytes
    for byte_i in range(input_len-3):
        flip_byte_afl(out_buff, byte_i)
        flip_byte_afl(out_buff, byte_i+1)
        flip_byte_afl(out_buff, byte_i+2)
        flip_byte_afl(out_buff, byte_i+3)
        yield out_buff
        flip_byte_afl(out_buff, byte_i)
        flip_byte_afl(out_buff, byte_i+1)
        flip_byte_afl(out_buff, byte_i+2)
        flip_byte_afl(out_buff, byte_i+3)


    # # flip two bits
    # # iterate over all bytes
    # for byte_i in range(input_len):

    #     # iterate over bits in the byte
    #     for bit_i in range(7):
    #         flip_bit(out_buff, 2, byte_i, bit_i)
    #         yield out_buff
    #         flip_bit(out_buff, 2, byte_i, bit_i)


    
    # # flip four bits
    # # iterate over all bytes
    # for byte_i in range(input_len):

    #     # iterate over bits in the byte
    #     for bit_i in range(5):
    #         flip_bit(out_buff, 4, byte_i, bit_i)
    #         yield out_buff
    #         flip_bit(out_buff, 4, byte_i, bit_i)



# must take bytearray
# @jit
# jit not working with byte arrays for some reason
def deterministic_edits_old(orig, out_buff):
    
    yield out_buff

    input_len = len(input)

    # flip single bits

    # iterate over all bytes
    for byte_i in range(input_len):

        # iterate over bits in the byte
        for bit_i in range(8):
            flip_bit(out_buff, 1, byte_i, bit_i)
            yield out_buff
            flip_bit(out_buff, 1, byte_i, bit_i)


    # flip two bits
    # iterate over all bytes
    for byte_i in range(input_len):

        # iterate over bits in the byte
        for bit_i in range(7):
            flip_bit(out_buff, 2, byte_i, bit_i)
            yield out_buff
            flip_bit(out_buff, 2, byte_i, bit_i)


    
    # flip four bits
    # iterate over all bytes
    for byte_i in range(input_len):

        # iterate over bits in the byte
        for bit_i in range(5):
            flip_bit(out_buff, 4, byte_i, bit_i)
            yield out_buff
            flip_bit(out_buff, 4, byte_i, bit_i)