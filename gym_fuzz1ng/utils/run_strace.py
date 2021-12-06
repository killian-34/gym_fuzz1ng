#!/usr/bin/env python

'''Script to run strace on a given binary or command, and store the output of syscalls in a pandas DataFrame'''

import sys
import subprocess
import pandas as pd
import numpy as np


__authors__ = ["Jackson Killian", "Susobhan Ghosh"]
__credits__ = ["Jackson Killian", "Susobhan Ghosh", "James Mickens"]
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "susobhan_ghosh@g.harvard.edu"
__status__ = "Development"


def run_strace(path, input, timeout=60):
	'''
	Runs strace on a given binary or command
	@path: Path of the binary or command
	@input: Input to be passed to the binary
	@timeout: Maximum time allowed to run the program with given input (in seconds)
	'''

	# Check if file is empty
	if path is None:
		print("File path is none, exiting.")
		sys.exit(1)

	# Build the command to run
	cmd_process = ["strace", "-c", "-qq", str(path)]
	if input is not None:
		cmd_process.append(str(input))

	# Store the output of strace and the command
	# result.stderr - strace output
	# result.stdout - command's stdout output
	result = subprocess.run(cmd_process, timeout=timeout, capture_output=True, text=True)
	lines = result.stderr.split("\n")

	# ignore lines until you find the strace output
	start = 0
	end = 0
	for i in range(len(lines)):
		if "time" in lines[i] and "seconds" in lines[i] and "calls" in lines[i]:
			start = i
		elif "100.00" in lines[i] and "total" in lines[i]:
			end = i

	# the strace output, by lines (skip the first two lines, and last two)
	strace_output = lines[start + 2: end - 1]

	# the strace headers, skip the "%" at the start
	strace_headers = lines[start][1:].split()

	# formats the strace output, and makes sure that the "errors" column is filled
	formatted_strace = []
	for i in strace_output:
		temp = i.split()
		if len(temp) != len(strace_headers):
			temp.insert(4, "0")
		formatted_strace.append(temp)
	
	# format the list of list into a dataframe
	strace_pd = pd.DataFrame(formatted_strace, columns=strace_headers)

	print(strace_pd)

def main():
	'''Call run_strace and log the output'''
	run_strace("ls", None)


if __name__ == '__main__':
	main()