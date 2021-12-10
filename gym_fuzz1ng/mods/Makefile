# Builds afl altered by our mod for external fuzzing as well as all of our
# targets.

# AFL

AFL_VERSION=2.52b
AFL_MOD=afl-$(AFL_VERSION)-mod
AFL_DIR=$(AFL_MOD)/afl-$(AFL_VERSION)

ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

AFL_FUZZ=$(ROOT_DIR)/$(AFL_DIR)/afl-fuzz
AFL_GCC=$(ROOT_DIR)/$(AFL_DIR)/afl-gcc
AFL_GXX=$(ROOT_DIR)/$(AFL_DIR)/afl-g++

$(AFL_DIR):
	@echo " **** Downloading afl and applying fuzz1ng mod"
	cd $(AFL_MOD)/ && wget http://lcamtuf.coredump.cx/afl/releases/afl-$(AFL_VERSION).tgz
	cd $(AFL_MOD)/ && tar xvf afl-$(AFL_VERSION).tgz
	cp -R $(AFL_MOD)/mod/* $(AFL_DIR)/

$(AFL_DIR)/afl-fuzz: $(AFL_DIR)
	@echo " **** Building afl with fuzz1ng mod"
	cd $(AFL_DIR)/ && make

# SIMPLE_2LADDER_SYSCALL

SIMPLE_2LADDER_SYSCALL_MOD=simple_2ladder_syscall-mod

$(SIMPLE_2LADDER_SYSCALL_MOD)/simple_2ladder_syscall_afl:
	@echo " **** Building simple_2ladder_syscall"
	cd $(SIMPLE_2LADDER_SYSCALL_MOD) && $(AFL_GCC) simple_2ladder_syscall_afl.c \
	  -o simple_2ladder_syscall_afl -lz -lm

# SIMPLE_BITS

SIMPLE_BITS_MOD=simple_bits-mod

$(SIMPLE_BITS_MOD)/simple_bits_afl:
	@echo " **** Building simple_bits_afl"
	cd $(SIMPLE_BITS_MOD) && $(AFL_GCC) simple_bits_afl.c \
	  -o simple_bits_afl -lz -lm

# ALL

all: $(AFL_DIR)/afl-fuzz \
  $(SIMPLE_2LADDER_SYSCALL_MOD)/simple_2ladder_syscall_afl \
  $(SIMPLE_BITS_MOD)/simple_bits_afl

clean:
	rm -rf $(AFL_DIR)
	rm -rf $(AFL_MOD)/afl-$(AFL_VERSION).tgz
	rm -rf $(SIMPLE_2LADDER_SYSCALL_MOD)/simple_2ladder_syscall_afl
	rm -rf $(SIMPLE_BITS_MOD)/simple_bits_afl
