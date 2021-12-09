import numpy as np
from collections import deque
from numba import jit
import time

ARITH_MAX = 4
DEV_MODE = True

INTERESTING_8 = [
  -128,          #/* Overflow signed 8-bit when decremented  */ \
  -1,            #/*                                         */ \
   0,            #/*                                         */ \
   1,            #/*                                         */ \
   16,           #/* One-off with common buffer size         */ \
   32,           #/* One-off with common buffer size         */ \
   64,           #/* One-off with common buffer size         */ \
   100,          #/* One-off with common buffer size         */ \
   127           #/* Overflow signed 8-bit when incremented  */
]

INTERESTING_16 = [
  -32768,        #/* Overflow signed 16-bit when decremented */ \
  -129,          #/* Overflow signed 8-bit                   */ \
   128,          #/* Overflow signed 8-bit                   */ \
   255,          #/* Overflow unsig 8-bit when incremented   */ \
   256,          #/* Overflow unsig 8-bit                    */ \
   512,          #/* One-off with common buffer size         */ \
   1000,         #/* One-off with common buffer size         */ \
   1024,         #/* One-off with common buffer size         */ \
   4096,         #/* One-off with common buffer size         */ \
   32767         #/* Overflow signed 16-bit when incremented */
]

INTERESTING_32 = [
  -2147483648,   #/* Overflow signed 32-bit when decremented */ \
  -100663046,    #/* Large negative number (endian-agnostic) */ \
  -32769,        #/* Overflow signed 16-bit                  */ \
   32768,        #/* Overflow signed 16-bit                  */ \
   65535,        #/* Overflow unsig 16-bit when incremented  */ \
   65536,        #/* Overflow unsig 16 bit                   */ \
   100663045,    #/* Large positive number (endian-agnostic) */ \
   2147483647    #/* Overflow signed 32-bit when incremented */
]

@jit
def flip_bit_afl(a, bit_i):
    a[bit_i >> 3] ^= (128 >> (bit_i & 7))

@jit
def flip_byte_afl(a, byte_i):
    a[byte_i] ^= 0xFF

def deterministic_bit_edit(in_buff, bit_i):
    flip_bit_afl(in_buff, bit_i)
    return in_buff


def random_edits(current_input, a):

    input_len = len(current_input)
    out_buff = current_input

    if a == 0:
        # flip single bits
        bit_i = np.random.randint(0, high = input_len << 3)
        flip_bit_afl(out_buff, bit_i)
        return out_buff
    
    if a == 1:
        # flip two bits
        bit_i = np.random.randint(0, high = (input_len << 3) - 1)
        flip_bit_afl(out_buff, bit_i)
        flip_bit_afl(out_buff, bit_i+1)
        return out_buff

    # snew = time.time()
    # print("Finished 2-bit flips: %s"%(snew-sprev))
    # sprev = snew

    # # flip four bits
    # for bit_i in range(input_len << 3 - 3):
    #     flip_bit_afl(out_buff, bit_i)
    #     flip_bit_afl(out_buff, bit_i+1)
    #     flip_bit_afl(out_buff, bit_i+2)
    #     flip_bit_afl(out_buff, bit_i+3)
    #     yield out_buff
    #     flip_bit_afl(out_buff, bit_i)
    #     flip_bit_afl(out_buff, bit_i+1)
    #     flip_bit_afl(out_buff, bit_i+2)
    #     flip_bit_afl(out_buff, bit_i+3)

    # snew = time.time()
    # print("Finished 4-bit flips: %s"%(snew-sprev))
    # sprev = snew


    # # flip byte
    # for byte_i in range(input_len):
    #     flip_byte_afl(out_buff, byte_i)
    #     yield out_buff
    #     flip_byte_afl(out_buff, byte_i)

    # snew = time.time()
    # print("Finished 1-byte flips: %s"%(snew-sprev))
    # sprev = snew

    # # flip 2 bytes
    # for byte_i in range(input_len-1):
    #     flip_byte_afl(out_buff, byte_i)
    #     flip_byte_afl(out_buff, byte_i+1)
    #     yield out_buff
    #     flip_byte_afl(out_buff, byte_i)
    #     flip_byte_afl(out_buff, byte_i+1)

    # snew = time.time()
    # print("Finished 2-byte flips: %s"%(snew-sprev))
    # sprev = snew

    # # flip 4 bytes
    # for byte_i in range(input_len-3):
    #     flip_byte_afl(out_buff, byte_i)
    #     flip_byte_afl(out_buff, byte_i+1)
    #     flip_byte_afl(out_buff, byte_i+2)
    #     flip_byte_afl(out_buff, byte_i+3)
    #     yield out_buff
    #     flip_byte_afl(out_buff, byte_i)
    #     flip_byte_afl(out_buff, byte_i+1)
    #     flip_byte_afl(out_buff, byte_i+2)
    #     flip_byte_afl(out_buff, byte_i+3)

    # snew = time.time()
    # print("Finished 4-byte flips: %s"%(snew-sprev))
    # sprev = snew

    # # 1-byte arithmetic
    # orig = np.copy(np.frombuffer(out_buff, dtype=np.uint8)) # need uint8 for correct overflow logic
    # for byte_i in range(input_len):
    #     for j in np.arange(1, ARITH_MAX, dtype=np.uint8):
    #         # import pdb; pdb.set_trace()
    #         out_buff[byte_i] = (orig[byte_i] + j) # need everything to be uint8 
    #         yield out_buff
    #         out_buff[byte_i] = (orig[byte_i] - j) # need uint8 for correct overflow logic
    #         yield out_buff
    #     out_buff[byte_i] = orig[byte_i]

    # snew = time.time()
    # print("Finished 1-byte arithmetic: %s"%(snew-sprev))
    # sprev = snew

    # # 2-byte arithmetic
    # NBYTES=2
    # for byte_i in range(input_len-1):

    #     orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy

    #     little = np.frombuffer(orig, dtype='<u%i'%NBYTES)
    #     big = np.frombuffer(orig, dtype='>u%i'%NBYTES)

    #     for j in range(1, ARITH_MAX):
    #         # little endian
    #         out_buff[byte_i:byte_i+NBYTES] = (little + j).tobytes()
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = (little - j).tobytes()
    #         yield out_buff

    #         out_buff[byte_i:byte_i+NBYTES] = (big + j).byteswap(inplace=True).tobytes()
    #         # out_buff[byte_i:byte_i+NBYTES] = (big + j)[0].to_bytes(NBYTES, byteorder='big')
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = (big - j).byteswap(inplace=True).tobytes()
    #         yield out_buff

    #     out_buff[byte_i:byte_i+NBYTES] = orig


    # snew = time.time()
    # print("Finished 2-byte arithmetic: %s"%(snew-sprev))
    # sprev = snew

    # # 4-byte arithmetic
    # NBYTES=4
    # for byte_i in range(input_len-3):

    #     orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy

    #     little = np.frombuffer(orig, dtype='<u%i'%NBYTES)
    #     big = np.frombuffer(orig, dtype='>u%i'%NBYTES)

    #     for j in range(1, ARITH_MAX):
    #         # little endian
    #         out_buff[byte_i:byte_i+NBYTES] = (little + j).tobytes()
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = (little - j).tobytes()
    #         yield out_buff

    #         out_buff[byte_i:byte_i+NBYTES] = (big + j).byteswap(inplace=True).tobytes()
    #         # out_buff[byte_i:byte_i+NBYTES] = (big + j)[0].to_bytes(NBYTES, byteorder='big')
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = (big - j).byteswap(inplace=True).tobytes()
    #         yield out_buff

    #     out_buff[byte_i:byte_i+NBYTES] = orig

    # snew = time.time()
    # print("Finished 4-byte arithmetic: %s"%(snew-sprev))
    # sprev = snew

    # # 1-byte interesting values
    # interesting_8 = np.array(INTERESTING_8, dtype=np.int8)
    # for byte_i in range(input_len):
    #     orig = out_buff[byte_i] # creates a copy
    #     for interesting in interesting_8:
    #         # import pdb;pdb.set_trace()
    #         out_buff[byte_i:byte_i+1] = interesting.tobytes()
    #         yield out_buff
    #     out_buff[byte_i] = orig


    # snew = time.time()
    # print("Finished 1-byte interesting values: %s"%(snew-sprev))
    # sprev = snew

    # # 2-byte interesting values
    # NBYTES=2
    # interesting_16 = np.array(INTERESTING_8+INTERESTING_16, dtype=np.int16)
    # interesting_16_big = interesting_16.byteswap()
    # for byte_i in range(input_len-1):

    #     orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy
    #     for j in range(len(interesting_16)):
    #         # import pdb;pdb.set_trace()
    #         out_buff[byte_i:byte_i+NBYTES] = interesting_16[j].tobytes()
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = interesting_16_big[j].tobytes()
    #         yield out_buff

    #     out_buff[byte_i:byte_i+NBYTES] = orig

    # snew = time.time()
    # print("Finished 2-byte interesting values: %s"%(snew-sprev))
    # sprev = snew

    # # 4-byte interesting values
    # NBYTES=4
    # interesting_32 = np.array(INTERESTING_8+INTERESTING_16+INTERESTING_32, dtype=np.int32)
    # interesting_32_big = interesting_32.byteswap()
    # for byte_i in range(input_len-3):

    #     orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy
    #     for j in range(len(interesting_32)):
    #         # import pdb;pdb.set_trace()
    #         out_buff[byte_i:byte_i+NBYTES] = interesting_32[j].tobytes()
    #         yield out_buff
    #         out_buff[byte_i:byte_i+NBYTES] = interesting_32_big[j].tobytes()
    #         yield out_buff

    #     out_buff[byte_i:byte_i+NBYTES] = orig


    # snew = time.time()
    # print("Finished 4-byte interesting values: %s"%(snew-sprev))
    # sprev = snew






# must take bytearray
# @jit
# jit not working with byte arrays for some reason
def deterministic_edits(orig, out_buff):
    
    yield out_buff

    input_len = len(orig)
    if DEV_MODE:
        input_len = 5

    s_determ = time.time()
    sprev = s_determ
    print("Starting deterministic edits!")
    
    # flip single bits
    for bit_i in range(input_len << 3):
        flip_bit_afl(out_buff, bit_i)
        yield out_buff
        flip_bit_afl(out_buff, bit_i)
    
    snew = time.time()
    print("Finished 1-bit flips: %s"%(snew-sprev))
    sprev = snew
    

    if not DEV_MODE:
        # flip two bits
        for bit_i in range(input_len << 3 - 1):
            flip_bit_afl(out_buff, bit_i)
            flip_bit_afl(out_buff, bit_i+1)
            yield out_buff
            flip_bit_afl(out_buff, bit_i)
            flip_bit_afl(out_buff, bit_i+1)

        snew = time.time()
        print("Finished 2-bit flips: %s"%(snew-sprev))
        sprev = snew

        

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

        snew = time.time()
        print("Finished 4-bit flips: %s"%(snew-sprev))
        sprev = snew


        # flip byte
        for byte_i in range(input_len):
            flip_byte_afl(out_buff, byte_i)
            yield out_buff
            flip_byte_afl(out_buff, byte_i)

        snew = time.time()
        print("Finished 1-byte flips: %s"%(snew-sprev))
        sprev = snew

        # flip 2 bytes
        for byte_i in range(input_len-1):
            flip_byte_afl(out_buff, byte_i)
            flip_byte_afl(out_buff, byte_i+1)
            yield out_buff
            flip_byte_afl(out_buff, byte_i)
            flip_byte_afl(out_buff, byte_i+1)

        snew = time.time()
        print("Finished 2-byte flips: %s"%(snew-sprev))
        sprev = snew

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

        snew = time.time()
        print("Finished 4-byte flips: %s"%(snew-sprev))
        sprev = snew

        # 1-byte arithmetic
        orig = np.copy(np.frombuffer(out_buff, dtype=np.uint8)) # need uint8 for correct overflow logic
        for byte_i in range(input_len):
            for j in np.arange(1, ARITH_MAX, dtype=np.uint8):
                # import pdb; pdb.set_trace()
                out_buff[byte_i] = (orig[byte_i] + j) # need everything to be uint8 
                yield out_buff
                out_buff[byte_i] = (orig[byte_i] - j) # need uint8 for correct overflow logic
                yield out_buff
            out_buff[byte_i] = orig[byte_i]

        snew = time.time()
        print("Finished 1-byte arithmetic: %s"%(snew-sprev))
        sprev = snew

        # 2-byte arithmetic
        NBYTES=2
        for byte_i in range(input_len-1):

            orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy

            little = np.frombuffer(orig, dtype='<u%i'%NBYTES)
            big = np.frombuffer(orig, dtype='>u%i'%NBYTES)

            for j in range(1, ARITH_MAX):
                # little endian
                out_buff[byte_i:byte_i+NBYTES] = (little + j).tobytes()
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = (little - j).tobytes()
                yield out_buff

                out_buff[byte_i:byte_i+NBYTES] = (big + j).byteswap(inplace=True).tobytes()
                # out_buff[byte_i:byte_i+NBYTES] = (big + j)[0].to_bytes(NBYTES, byteorder='big')
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = (big - j).byteswap(inplace=True).tobytes()
                yield out_buff

            out_buff[byte_i:byte_i+NBYTES] = orig


        snew = time.time()
        print("Finished 2-byte arithmetic: %s"%(snew-sprev))
        sprev = snew

        # 4-byte arithmetic
        NBYTES=4
        for byte_i in range(input_len-3):

            orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy

            little = np.frombuffer(orig, dtype='<u%i'%NBYTES)
            big = np.frombuffer(orig, dtype='>u%i'%NBYTES)

            for j in range(1, ARITH_MAX):
                # little endian
                out_buff[byte_i:byte_i+NBYTES] = (little + j).tobytes()
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = (little - j).tobytes()
                yield out_buff

                out_buff[byte_i:byte_i+NBYTES] = (big + j).byteswap(inplace=True).tobytes()
                # out_buff[byte_i:byte_i+NBYTES] = (big + j)[0].to_bytes(NBYTES, byteorder='big')
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = (big - j).byteswap(inplace=True).tobytes()
                yield out_buff

            out_buff[byte_i:byte_i+NBYTES] = orig

        snew = time.time()
        print("Finished 4-byte arithmetic: %s"%(snew-sprev))
        sprev = snew

        # 1-byte interesting values
        interesting_8 = np.array(INTERESTING_8, dtype=np.int8)
        for byte_i in range(input_len):
            orig = out_buff[byte_i] # creates a copy
            for interesting in interesting_8:
                # import pdb;pdb.set_trace()
                out_buff[byte_i:byte_i+1] = interesting.tobytes()
                yield out_buff
            out_buff[byte_i] = orig


        snew = time.time()
        print("Finished 1-byte interesting values: %s"%(snew-sprev))
        sprev = snew

        # 2-byte interesting values
        NBYTES=2
        interesting_16 = np.array(INTERESTING_8+INTERESTING_16, dtype=np.int16)
        interesting_16_big = interesting_16.byteswap()
        for byte_i in range(input_len-1):

            orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy
            for j in range(len(interesting_16)):
                # import pdb;pdb.set_trace()
                out_buff[byte_i:byte_i+NBYTES] = interesting_16[j].tobytes()
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = interesting_16_big[j].tobytes()
                yield out_buff

            out_buff[byte_i:byte_i+NBYTES] = orig

        snew = time.time()
        print("Finished 2-byte interesting values: %s"%(snew-sprev))
        sprev = snew

        # 4-byte interesting values
        NBYTES=4
        interesting_32 = np.array(INTERESTING_8+INTERESTING_16+INTERESTING_32, dtype=np.int32)
        interesting_32_big = interesting_32.byteswap()
        for byte_i in range(input_len-3):

            orig = out_buff[byte_i:byte_i+NBYTES] # creates a copy
            for j in range(len(interesting_32)):
                # import pdb;pdb.set_trace()
                out_buff[byte_i:byte_i+NBYTES] = interesting_32[j].tobytes()
                yield out_buff
                out_buff[byte_i:byte_i+NBYTES] = interesting_32_big[j].tobytes()
                yield out_buff

            out_buff[byte_i:byte_i+NBYTES] = orig


        snew = time.time()
        print("Finished 4-byte interesting values: %s"%(snew-sprev))
        sprev = snew

