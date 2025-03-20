import time
import argparse
import resource
import subprocess


def get_limit_resource(time_limit, memory_limit):
    def limit_resources():
        if time_limit is not None:
            resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit + 5))

        if memory_limit is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (memory_limit * 1024 * 1024, memory_limit * 1024 * 1024),
            )

    return limit_resources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, nargs="+")
    parser.add_argument("--memory-limit", type=int, default=None)
    parser.add_argument("--time-limit", type=int, default=None)
    args, remaining = parser.parse_known_args()

    fn = get_limit_resource(args.time_limit, args.memory_limit)
    start = time.perf_counter()
    popen = subprocess.Popen(
        args=args.command + remaining,
        preexec_fn=fn,
    )
    out, err = popen.communicate()
    end = time.perf_counter()
    print("Execution time: {}s".format(end - start))