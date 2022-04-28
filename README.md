# hpc-codes

1.  print “Hello World" using MPI
#include <iostream> 
#include "mpi.h" 
int main(int* argc, char* argv) 
{ 
int commsize; 
int rank; 
MPI_Init(NULL, NULL); 
MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
MPI_Comm_size(MPI_COMM_WORLD, &commsize); 
printf("Hello World from Process no. %d\n", rank); 
MPI_Finalize(); 
return 0; 
} 

2. Write a C program to find Sum of an array using MPI
#include "mpi.h" 
#include <stdio.h> 
#include <stdlib.h> 
#define n 10 
int a[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; 
// Temporary array for other processes 
int b[1000]; 
int main(int argc, char* argv[]) 
{ 
int process_id, no_of_process, 
elements_per_process, 
n_elements_recieved; 
MPI_Status status; 
MPI_Init(&argc, &argv); 
MPI_Comm_rank(MPI_COMM_WORLD, &process_id); 
MPI_Comm_size(MPI_COMM_WORLD, &no_of_process); 
// For process 0 // master process
if (process_id == 0) { 
int index, i; 
elements_per_process = n / no_of_process; 
if (no_of_process > 1) {  // check if more than 1 processes are run
for (i = 1; i < no_of_process - 1; i++) {       // distributes the portion of array to child processes to calculate partial sums

index = i * elements_per_process; 
MPI_Send(&elements_per_process, 1, MPI_INT, i, 0, MPI_COMM_WORLD); MPI_Send(&a[index], elements_per_process, MPI_INT, i, 0,  
MPI_COMM_WORLD); 
} 
// last process adds remaining elements
 
index = i * elements_per_process; 
int elements_left = n - index; 
MPI_Send(&elements_left, 1, MPI_INT, i, 0, MPI_COMM_WORLD); 
MPI_Send(&a[index], elements_left, MPI_INT, i, 0, MPI_COMM_WORLD); } 
// sum by process 0  // master process add its own sub array
int sum = 0; 
for (i = 0; i < elements_per_process; i++) 
sum += a[i]; 
printf("Sum by process %d = %d\n", process_id, sum); 
// partial sums from other processes  // collects partial sums from other processes
int tmp; 
for (i = 1; i < no_of_process; i++) { 
MPI_Recv(&tmp, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status); int sender = status.MPI_SOURCE; 
sum += tmp; 
} 
// prints the final sum of array // prints the final sum of array 
printf("Final Sum of array is : %d\n", sum); 
} 
// Other processes //Slave Processes
Else // stores the received array segment
        // in local array b
 { 
MPI_Recv(&n_elements_recieved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); MPI_Recv(&b, n_elements_recieved, MPI_INT, 0, 0, MPI_COMM_WORLD, &status); 
int partial_sum = 0; 
for (int i = 0; i < n_elements_recieved; i++) 
partial_sum += b[i]; 
printf("Sum by process %d = %d\n", process_id, partial_sum); 
// sends the partial sum to the root processMPI_Send(&partial_sum, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); 
} 
MPI_Finalize(); 
return 0; 
}

5. Write a multithreaded program that generates the Fibonacci  series using the pThreads library.
#include <iostream> 
#include <unistd.h> 
#include <pthread.h> 
using namespace std; 
pthread_t thread_id; 
int *data; 
void *threadfunction1(void *args) 
{ 
sleep(1);  //uses unistd.h
cout << "Thread called!" << endl; 
cout << endl; 
cout << "The value of Thread ID is: " << pthread_self() << endl; // self()  returns the thread ID of the current thread 
cout << endl; 
int n = *((int *)args); 
data = new int[n]; 
if (n > 0) 
data[0] = 0; 
if (n > 1) 
data[1] = 1; 
if (n > 2) 
{ 
for (int i = 2; i < n; i++) 
data[i] = data[i - 1] + data[i - 2];
 
} 
pthread_detach(pthread_self()); // Detach() frees the resources of pthread_self pthread_exit(NULL); // Exit thread, line after this will not print cout << "This will not be printed!" << endl; 
} 
int main(int argc, char *argv[]) 
{ 
cout << "\tMain function starts..." << endl; 
cout << endl; 
int n = atoi(argv[1]);  //converts string to an integer stdlib/h
pthread_create(&thread_id, NULL, threadfunction1, &n); // Thread Create function pthread_join(thread_id, NULL); // Waiting for function  to execute 
cout << "First " << n << " fibonacci numbers are: " << endl; 
for (int i = 0; i < n; i++) 
cout << data[i] << " "; 
cout << endl; 
pthread_exit(NULL); // exit thread 
exit(0); // exit process 
} 

6. Implement a program for Process Synchronization by mutex  locks using pThreads.
#include <iostream> 
#include <unistd.h> 
#include <pthread.h> 
using namespace std; 
pthread_mutex_t lock;  //Locks a mutex object, which identifies a mutex.
int n = 1000; 
void *threadfunction1(void *args) 
{ 
sleep(1); 
cout << "Increment Thread called!\n"; 
cout << endl; 
pthread_mutex_lock(&lock);  // Locks a mutex object, which identifies a mutex. If the mutex is already locked by another thread, the thread waits for the mutex to become available
n = n + 1; // CRITICAL SECTION 
cout << n << endl; 
pthread_mutex_unlock(&lock); 
pthread_detach(pthread_self()); 
pthread_exit(NULL); 
cout << "This will not be printed!" << endl; 
} 
void *threadfunction2(void *args) 
{ 
sleep(1); 
cout << "Decrement Thread called!\n"; 
cout << endl; 
pthread_mutex_lock(&lock); 
n = n - 1; // CRITICAL SECTION 
cout << n << endl; 
pthread_mutex_unlock(&lock); 
pthread_detach(pthread_self()); 
pthread_exit(NULL); 
cout << "This will not be printed!" << endl;
} 
int main(int argc, char *argv[]) 
{ 
if (pthread_mutex_init(&lock, NULL) != 0) //Creates a mutex, referenced by mutex
{ 
printf("\n mutex init has failed\n"); 
return 1; 
} 
cout << "\tMain function starts..." << endl; 
cout << endl; 
cout << "The initial value of shared variable is: " << n << endl; 
pthread_t thread_id1;  //pthread_t is the data type used to uniquely identify a thread
pthread_t thread_id2; 
pthread_create(&thread_id1, NULL, threadfunction1, NULL); pthread_create(&thread_id2, NULL, threadfunction2, NULL); 
pthread_join(thread_id1, NULL); 
pthread_join(thread_id2, NULL); 
cout << "The final value of shared variable is: " << n << endl; 
pthread_mutex_destroy(&lock); 
pthread_exit(NULL); // exit thread 
exit(0); // exit process 
} 

7. Implement a C program which demonstrates how to  "multitask" openmp
#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h> 
int fib(int n) 
{ 
int res; 
if (n == 0 or n == 1) 
res = n; 
else 
{ 
int a, b; 
#pragma omp task shared(a)  //Use the task pragma when you want to identify a block of code to be executed in parallel with the code outside the task region.
a = fib(n - 1); 
#pragma omp task shared(b) 
b = fib(n - 2); 
#pragma omp taskwait // Use the taskwait pragma to specify a wait for child tasks to be completed that are generated by the current task.
res = a + b; 
} 
printf("%d th Fibonacci task calculated by thread %d\n", n,  omp_get_thread_num()); 
return res; //The omp_get_thread_num routine returns the thread number, within the current team, of the calling thread.
} 
int main(int argc, char *argv[]) 
{ 
#pragma omp parallel //explicitly instructs the compiler to parallelize the chosen block of code.
#pragma omp single //The omp single directive identifies a section of code that must be run by a single available thread.
{ 
int n = atoi(argv[1]); //stdlib
printf("\nThe %d th Fibonacci Number = %d\n", n, fib(n)); } 
 return 0; 
}

8.  Implement a C program which demonstrates the default, static  and dynamic methods of "scheduling" loop iterations in OpenMP
#include <omp.h> 
#include <stdio.h> 
#include <stdlib.h> 
int main() 
{ int i, N = 10, THREAD_COUNT = 3, CHUNK_SIZE = 3; 
printf("Default Scheduling\n"); 
#pragma omp parallel for num_threads(THREAD_COUNT)  //num_thread is numb of threads in a parallel region 
for (i = 0; i < N; i++) 
printf("ThreadID: %d, iteration: %d\n", omp_get_thread_num(), i); 

printf("\nStatic Scheduling\n"); 
#pragma omp parallel for num_threads(THREAD_COUNT) schedule(static, CHUNK_SIZE) for (i = 0; i < N; i++) 
printf("ThreadID: %d, iteration: %d\n", omp_get_thread_num(), i); 
printf("\nDynamic Scheduling\n"); 
#pragma omp parallel for num_threads(THREAD_COUNT) schedule(dynamic, CHUNK_SIZE) for (i = 0; i < N; i++) 
printf("ThreadID: %d, iteration: %d\n", omp_get_thread_num(), i); return 0; 
}

3. Write a parallel program for parallel implementation of matrix  multiplication using MPI.
#include<stdio.h> 
#include<iostream> 
#include "mpi.h" 
#define NUM_ROWS_A 8 
#define NUM_COLUMNS_A 10 
#define NUM_ROWS_B 10 
#define NUM_COLUMNS_B 8 
#define MASTER_TO_SLAVE_TAG 1 //tag for messages sent from master to slaves #define SLAVE_TO_MASTER_TAG 4 //tag for messages sent from slaves to master 
void create_matrix();  
void printArray(); 
int rank;  
int size;  
int i, j, k;  
double A[NUM_ROWS_A][NUM_COLUMNS_A];  
double B[NUM_ROWS_B][NUM_COLUMNS_B];  
double result[NUM_ROWS_A][NUM_COLUMNS_B];  
int low_bound; //low bound of the number of rows of [A] allocated to a slave int upper_bound; //upper bound of the number of rows of [A] allocated to a slave int portion; //portion of the number of rows of [A] allocated to a slave MPI_Status status; // store status of a MPI_Recv 
MPI_Request request; //capture request of a MPI_Send 
int main(int argc, char* argv[]) 
{ 
MPI_Init(&argc, &argv);  
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
MPI_Comm_size(MPI_COMM_WORLD, &size);  
if (rank == 0)  
{ // master process 
create_matrix(); 
for (i = 1; i < size; i++)  
{ 
portion = (NUM_ROWS_A / (size - 1)); // portion without master 
low_bound = (i - 1) * portion; 
if (((i + 1) == size) && ((NUM_ROWS_A % (size - 1)) != 0))  
{//if rows of [A] cannot be equally divided among slaves 
upper_bound = NUM_ROWS_A; //last slave gets all the remaining rows } 
else {

upper_bound = low_bound + portion; //rows of [A] are equally  
divisable among slaves 
} 
MPI_Send(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG,  
MPI_COMM_WORLD); 
MPI_Send(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1,  
MPI_COMM_WORLD); 
MPI_Send(&A[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_A,  MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD); 
} 
} 
//broadcast [B] to all the slaves 
MPI_Bcast(&B, NUM_ROWS_B * NUM_COLUMNS_B, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
/* Slave process*/ 
if (rank > 0)  
{ 
MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD,  &status); 
MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1,  
MPI_COMM_WORLD, &status); 
MPI_Recv(&A[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_A,  MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status); 
printf("Process %d calculating for rows %d to %d of Matrix A\n", rank,  low_bound, upper_bound); 
for (i = low_bound; i < upper_bound; i++)  
{ 
for (j = 0; j < NUM_COLUMNS_B; j++)  
{ 
for (k = 0; k < NUM_ROWS_B; k++)  
{ 
result[i][j] += (A[i][k] * B[k][j]); 
} 
} 
} 
MPI_Send(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD); MPI_Send(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1,  
MPI_COMM_WORLD); 
MPI_Send(&result[low_bound][0], (upper_bound - low_bound) * NUM_COLUMNS_B,  MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD); 
} 
/* master gathers processed work*/ 
if (rank == 0) { 
for (i = 1; i < size; i++) { 
MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD,  &status); 
MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1,  
MPI_COMM_WORLD, &status);

MPI_Recv(&result[low_bound][0], (upper_bound - low_bound) *  
NUM_COLUMNS_B, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status); } 
printArray(); 
} 
MPI_Finalize();  
return 0; 
} void create_matrix() 
{ 
for (i = 0; i < NUM_ROWS_A; i++) { 
for (j = 0; j < NUM_COLUMNS_A; j++) { 
A[i][j] = i + j; 
} 
} 
for (i = 0; i < NUM_ROWS_B; i++) { 
for (j = 0; j < NUM_COLUMNS_B; j++) { 
B[i][j] = i * j; 
} 
} 
} void printArray() 
{ 
printf("The matrix A is: \n"); 
for (i = 0; i < NUM_ROWS_A; i++) { 
printf("\n"); 
for (j = 0; j < NUM_COLUMNS_A; j++) 
printf("%8.2f ", A[i][j]); // 8 reserved space rounded to 2 spaces will leave space if 8 is not utilised completely
} 
printf("\n\n\n"); 
printf("The matrix B is: \n"); 
for (i = 0; i < NUM_ROWS_B; i++) { 
printf("\n"); 
for (j = 0; j < NUM_COLUMNS_B; j++) 
printf("%8.2f ", B[i][j]); 
} 
printf("\n\n\n"); 
printf("The result matrix is: \n"); 
for (i = 0; i < NUM_ROWS_A; i++) { 
printf("\n"); 
for (j = 0; j < NUM_COLUMNS_B; j++) 
printf("%8.2f ", result[i][j]); 
} 
printf("\n\n"); 
}

4. Write a C program to implement the Quick Sort Algorithm  using MPI.
#include "mpi.h" 
#include <iostream> 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
using namespace std; 
#define ARRAY_SIZE 20 
void swap(int* arr, int i, int j) 
{ 
int t = arr[i]; 
arr[i] = arr[j]; 
arr[j] = t; 
} 
void quicksort(int* arr, int start, int end) 
{ 
int pivot, index; 
if (end <= 1) 
return; 
pivot = arr[start + end / 2]; 
swap(arr, start, start + end / 2); 
index = start; 
for (int i = start + 1; i < start + end; i++) { 
if (arr[i] < pivot) { 
index++; 
swap(arr, i, index); 
} 
} 
swap(arr, start, index); 
quicksort(arr, start, index - start); 
quicksort(arr, index + 1, start + end - index - 1); 
} 
int* merge(int* arr1, int n1, int* arr2, int n2) 
{ 
int* result = (int*)malloc((n1 + n2) * sizeof(int)); 
int i = 0; 
int j = 0; 
int k;
 
for (k = 0; k < n1 + n2; k++) { 
if (i >= n1) { 
result[k] = arr2[j]; 
j++; 
} 
else if (j >= n2) { 
result[k] = arr1[i]; 
i++; 
} 
else if (arr1[i] < arr2[j]) { 
result[k] = arr1[i]; 
i++; 
} 
else { 
result[k] = arr2[j]; 
j++; 
} 
} 
return result; 
} 
int main(int argc, char* argv[]) 
{ 
int number_of_elements; 
int* data = NULL; 
int chunk_size, own_chunk_size; 
int* chunk; 
MPI_Status status; 
int number_of_process, rank_of_process; 
MPI_Init(&argc, &argv); 
MPI_Comm_size(MPI_COMM_WORLD, &number_of_process); MPI_Comm_rank(MPI_COMM_WORLD, &rank_of_process); 
if (rank_of_process == 0)  
{ 
number_of_elements = ARRAY_SIZE; 
data = (int*)malloc(number_of_elements * sizeof(int)); //dynamically allocate block of memory 
for (int i = 0; i < number_of_elements; i++) 
data[i] = ARRAY_SIZE - i; 
printf("The initial array is : \n"); 
for (int i = 0; i < number_of_elements; i++) 
printf("%d ", data[i]); 
printf("\n"); 
}

// Blocks all process until reach this point 
MPI_Barrier(MPI_COMM_WORLD); 
// BroadCast the Size to all the process from root process 
MPI_Bcast(&number_of_elements, 1, MPI_INT, 0, MPI_COMM_WORLD); 
// Computing chunk size 
if (number_of_elements % number_of_process == 0) 
chunk_size = number_of_elements / number_of_process; 
else 
chunk_size = (number_of_elements / (number_of_process - 1)); 
// chunk array 
chunk = (int*)malloc(chunk_size * sizeof(int)); 
// Scatter the chuck size data to all process 
MPI_Scatter(data, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0,  MPI_COMM_WORLD); 
free(data); 
data = NULL; 
// Compute size of own chunk and then sort them using quick sort own_chunk_size = (number_of_elements >= chunk_size * (rank_of_process + 1)) ? chunk_size : (number_of_elements - 
chunk_size * rank_of_process); 
printf("The process %d sorted the following array: \n", rank_of_process); for (int i = 0;i < own_chunk_size;i++) 
printf("%d ", chunk[i]); 
printf("\n"); 
quicksort(chunk, 0, own_chunk_size); 
for (int step = 1; step < number_of_process; step = 2 * step) { 
if (rank_of_process % (2 * step) != 0)  
{ 
MPI_Send(chunk, own_chunk_size, MPI_INT, rank_of_process - step, 0,  MPI_COMM_WORLD); 
break; 
} if (rank_of_process + step < number_of_process)  
{ 
int received_chunk_size 
= (number_of_elements 
>= chunk_size 
* (rank_of_process + 2 * step)) 
? (chunk_size * step)

: (number_of_elements 
- chunk_size 
* (rank_of_process + step)); 
int* chunk_received; 
chunk_received = (int*)malloc(received_chunk_size * sizeof(int)); 
MPI_Recv(chunk_received, received_chunk_size, MPI_INT, rank_of_process +  step, 0, 
MPI_COMM_WORLD, &status); 
data = merge(chunk, own_chunk_size, chunk_received,  
received_chunk_size); 
free(chunk); 
free(chunk_received); 
chunk = data; 
own_chunk_size = own_chunk_size + received_chunk_size; 
} 
} 
if (rank_of_process == 0) 
{ 
printf("The Sorted array is: \n"); 
for (int i = 0; i < number_of_elements; i++)  
printf("%d ", chunk[i]); 
} MPI_Finalize(); 
return 0; 
}
