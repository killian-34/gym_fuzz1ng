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

void process_data(char* buffer, size_t size)
{
  struct stat st = {0};
  char PATH[DATA_SIZE_MAX] = "/tmp/";

  int spaceflag = 0;

  // checks if the file has at one string only separated by a space
  for(int i = 0; i < size; i++)
    if(isspace(buffer[i]))
      spaceflag++;

  if(spaceflag != 1)
    exit(-1);

  for(int i = 0; !isspace(buffer[i]) && i < size; i++)
  {
    if(buffer[i] == buffer[size - i - 1])
    {
      char extension[2] = {buffer[i], '/'};
      strncat(PATH, extension, 2);

      if(stat(PATH, &st) == -1)
        mkdir(PATH, 0700);
    }
    else
    {
      if (stat(PATH, &st) != -1)
        rmdir(PATH);
      printf("%s\n", buffer);
      printf("Failure\n");
      exit(-1);
    }
  }
  
  char *filename = "boom";
  strncat(PATH, filename, 4);

  int fd2 = open(PATH, O_RDWR|O_CREAT, 0777);

  if (fd2 != -1) {
      close(fd2);
  }
  printf("Success\n");
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

  process_data(buffer, st.st_size);

  return -1;
}
