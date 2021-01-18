import multiprocessing
import time

import psutil
from psutil._common import bytes2human
import pynvml


if __name__ == '__main__':
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    process_num = multiprocessing.cpu_count()
    total_gpu = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    total_mem = start_mem_used = psutil.virtual_memory().total
    summary = ''
    image_num = 50

    summary += '## Test Environment\n'
    summary += f'- total process numbers: {process_num}\n'
    summary += f'- total memory: {bytes2human(total_mem)}\n'
    summary += f'- total gpu memory: {bytes2human(total_gpu)}\n'

    print(summary)

    summary += '## Target\n'
    summary += f'- image numbers: {image_num}\n'
    summary += '## Result\n'

    for device in ['cpu', 'gpu']:
        summary += f'### {device}\n'
        summary += '|process|mem used|gpu mem used|total time used|time per image|other mem used|other gpu mem used|\n'
        summary += '|:-:|:-:|:-:|:-:|:-:|:-:|:-:|\n'
        for process in range(1, 2):
            start_gpu_used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            start_mem_used = psutil.virtual_memory().used

            print(f'process used: {process}')
            print(f'start mem used: {bytes2human(start_mem_used)}')
            print(f'start gpu used: {bytes2human(start_gpu_used)}')

            start_time = time.time()

            try:
                if device == 'cpu':   # load model time.
                    from tools.multiprocessing_test_cpu import main
                    max_mem_used, max_gpu_used = main(process)
                else:
                    from tools import multiprocessing_test_gpu
                    max_mem_used, max_gpu_used = multiprocessing_test_gpu.main(process)

                print(f'total time: {time.time() - start_time :.2f}s')
                print(f'used mem: {bytes2human(max_mem_used - start_mem_used)}')
                print(f'used gpu: {bytes2human(max_gpu_used - start_gpu_used)}')

                end_time = time.time()
                summary += f'|{process}|{bytes2human(max_mem_used - start_mem_used)}|{bytes2human(max_gpu_used - start_gpu_used)}|{end_time - start_time :.2f}s|{(end_time - start_time) / image_num :.2f}s|{bytes2human(start_mem_used)}|{bytes2human(start_gpu_used)}|\n'
            except RuntimeError:
                break

    print(summary)
    with open('summary.md', 'w', encoding='utf-8') as f:
        f.write(summary)
