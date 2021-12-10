/*
 * Toy example code for invoking syscalls at branching instructions
 */

#include <stdio.h>
#include <stdlib.h> // malloc
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <ctype.h>

#define DATA_SIZE_MAX 0x00200000


void do_reads(int upper){
  for (int j=0; j<upper; j++){
    int i = read(0, 0, 0);
  }
}

void do_writes(int upper){
  for (int j=0; j<upper; j++){
    int i = write(0, 0, 0);
  }
}

void two_ladder(char* buffer, size_t size)
{
  // go down the read ladder
  if(buffer[size/2] > 57){ // this is ascii 9
    int i = read(0, 0, 0);
    // printf("reading file %i\n",i);

    int BUFF_NUM = 0;

    if (buffer[BUFF_NUM] < 4){
      do_reads(4);
      // printf(" < 4 read %i\n",i);
    }
    if (4 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 42){
      do_reads(8);
      // printf("5 < x < 10 read %i\n",i);
    }
    if (42 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 120){
      do_reads(12);
      // printf("42 < x < 82 read %i\n",i);
    }
    if (120 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 256){
      do_reads(16);
      // printf("120 < x < 150 read %i\n",i);
    }
  }

  // go down the write ladder 
  else{
    int i = write(0, 0, 0);
    // printf("writing file %i\n",i);

    int BUFF_NUM = size - 1;

  
    if (buffer[BUFF_NUM] < 4){
      do_writes(4);
      // printf(" < 4 write %i\n",i);
    }
    if (4 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 42){
      do_writes(8);
      // printf("5 < x < 10 write %i\n",i);
    }
    // H is a hit
    if (42 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 120){
      do_writes(12);
      // printf("42 < x < 82 write %i\n",i);
    }
    // z is a hit
    if (120 <= buffer[BUFF_NUM] && buffer[BUFF_NUM] < 256){
      do_writes(16);
      // printf("120 < x < 150 write %i\n",i);
    }
  }
  
}


int main(int argc, char **argv)
{
  int fd;
  struct stat st;

  char buffer[DATA_SIZE_MAX];

  if (argc < 2) {
    printf("usage: simple_syscall_afl <file_in>\n");
    exit(-2);
  }

  if ((fd = open(argv[1], O_RDONLY)) < 0) {
    printf("error opening file");
    exit(-1);
  }

  if (fstat (fd, &st) < 0) {
    printf("stating file");
    exit(-1);
  }

  if (read(fd, buffer, st.st_size) < 0) {
    printf("error reading file");
    exit(-1);
  }

  printf("file size %li\n",st.st_size);
  // process_data(buffer, st.st_size);
  two_ladder(buffer, st.st_size);
  
  close(fd);

  return -1;
}
