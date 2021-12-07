/*
 * An adaptation of the code written by Nathan Voss <njvoss99@gmail.com> to
 * afl for use with gym_fuzz1ng.
 */
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv)
{
  u_int8_t a = 0;
  printf("%i\n",a);
  a = -127;
  printf("%i\n",a);

  return 0;
}
