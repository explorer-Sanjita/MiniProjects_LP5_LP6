use mpi::traits::*;
use rand::Rng;
use std::{
    fs::File,
    io::Write,
    time::{Duration, Instant},
    usize,
};

fn quicksort(arr: &mut [i32]) {
    if arr.len() <= 1 {
        return;
    }
    let pivot_index = partition(arr);
    quicksort(&mut arr[0..pivot_index]);
    quicksort(&mut arr[pivot_index + 1..]);
}

fn partition(arr: &mut [i32]) -> usize {
    let len = arr.len();
    let pivot = arr[len - 1];
    let mut i = 0;
    for j in 0..len - 1 {
        if arr[j] < pivot {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, len - 1);
    i
}

// Merge two sorted arrays into one
fn merge(left: &[i32], right: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut left_iter = left.iter();
    let mut right_iter = right.iter();

    let mut left_next = left_iter.next();
    let mut right_next = right_iter.next();

    while left_next.is_some() && right_next.is_some() {
        if left_next.unwrap() <= right_next.unwrap() {
            result.push(*left_next.unwrap());
            left_next = left_iter.next();
        } else {
            result.push(*right_next.unwrap());
            right_next = right_iter.next();
        }
    }

    // Append remaining elements
    while let Some(val) = left_next {
        result.push(*val);
        left_next = left_iter.next();
    }

    while let Some(val) = right_next {
        result.push(*val);
        right_next = right_iter.next();
    }

    result
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_rank = 0;

    // Define data sizes in the requested progression
    let data_sizes = vec![
        10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000,
        1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000,
    ];

    if rank == root_rank {
        // Create CSV file and write header
        let mut file = File::create("quicksort_benchmark.csv").expect("Failed to create file");
        writeln!(file, "id,data_size,parallel_qs_in_ms,sequential_qs_in_ms")
            .expect("Failed to write header");

        println!("Starting benchmarks for various data sizes...");
        println!("id,data_size,parallel_qs_in_ms,sequential_qs_in_ms");
    }

    // Barrier to ensure all processes start together
    world.barrier();

    // Run benchmark for each data size
    for (id, &total_size) in data_sizes.iter().enumerate() {
        let chunk_size = total_size / size as usize;
        // Ensure each process gets at least one element
        let adjusted_chunk_size = std::cmp::max(chunk_size, 1);

        let mut local_data = vec![0; adjusted_chunk_size];
        let mut full_data = Vec::new();

        let overall_parallel_start = Instant::now();
        let mut parallel_duration = Duration::from_secs(0);
        let mut sequential_duration = Duration::from_secs(0);

        if rank == root_rank {
            // Generate random data
            let mut rng = rand::thread_rng();
            full_data = (0..total_size).map(|_| rng.gen_range(0..100000)).collect();

            // For very small sizes, adjust distribution
            if size as usize > total_size {
                // If we have more processes than elements, only root process works
                local_data = full_data.clone();
            } else {
                // Distribute chunks to other processes
                for i in 1..size {
                    let start = i as usize * adjusted_chunk_size;
                    let end = if i as usize * adjusted_chunk_size >= total_size {
                        total_size
                    } else if i == size - 1 {
                        total_size
                    } else {
                        (i as usize + 1) * adjusted_chunk_size
                    };

                    if start < end && start < total_size {
                        let data_chunk = &full_data[start..end.min(total_size)];
                        world.process_at_rank(i).send(data_chunk);
                    } else {
                        // Send empty chunk if we're out of bounds
                        let empty_chunk: Vec<i32> = Vec::new();
                        world.process_at_rank(i).send(&empty_chunk[..]);
                    }
                }

                // Root process takes its own chunk
                let end_idx = std::cmp::min(adjusted_chunk_size, total_size);
                local_data = full_data[0..end_idx].to_vec();
            }
        } else {
            // Non-root processes receive their chunks
            let (received_chunk, _) = world.process_at_rank(root_rank).receive_vec::<i32>();
            local_data = received_chunk;
        }

        // Each process sorts its local chunk, if it has data
        if !local_data.is_empty() {
            quicksort(&mut local_data);
        }

        if rank == root_rank {
            // Collect and merge sorted chunks for parallel algorithm
            let mut sorted_data = local_data;

            for i in 1..size {
                let (received_chunk, _) = world.process_at_rank(i).receive_vec::<i32>();
                if !received_chunk.is_empty() {
                    sorted_data = merge(&sorted_data, &received_chunk);
                }
            }

            parallel_duration = overall_parallel_start.elapsed();

            // Run sequential version
            let mut sequential_data = full_data.clone();
            let sequential_start = Instant::now();
            quicksort(&mut sequential_data);
            sequential_duration = sequential_start.elapsed();

            // Verify correctness
            let is_parallel_sorted = sorted_data.windows(2).all(|w| w[0] <= w[1]);
            let is_sequential_sorted = sequential_data.windows(2).all(|w| w[0] <= w[1]);

            if !is_parallel_sorted || !is_sequential_sorted {
                println!("Warning: Sort verification failed for size {}", total_size);
            }

            // Convert to milliseconds
            let parallel_ms = parallel_duration.as_micros() as f64 / 1000.0;
            let sequential_ms = sequential_duration.as_micros() as f64 / 1000.0;

            // Write to CSV file
            let mut file = File::options()
                .append(true)
                .open("quicksort_benchmark.csv")
                .expect("Failed to open file");

            writeln!(
                file,
                "{},{},{:.3},{:.3}",
                id + 1,
                total_size,
                parallel_ms,
                sequential_ms
            )
            .expect("Failed to write data to CSV");

            // Also print to console for monitoring
            println!(
                "{},{},{:.3},{:.3}",
                id + 1,
                total_size,
                parallel_ms,
                sequential_ms
            );
        } else {
            // Send sorted data back to root
            world.process_at_rank(root_rank).send(&local_data[..]);
        }

        // Wait for all processes to finish before starting next benchmark
        world.barrier();
    }

    if rank == root_rank {
        println!("Benchmark completed. Results saved to quicksort_benchmark.csv");
    }
}
