import subprocess
import psutil
import time
import datetime
import threading
import os


class ResourceMonitor:
    def __init__(self, log_directory, interval=60):
        self.interval = interval
        self.stop_event = threading.Event()
        self.log_directory = log_directory
        self.log_file = os.path.join(log_directory, f"resource_usage_{int(time.time())}.log")
        os.makedirs(log_directory, exist_ok=True)

    def get_gpu_info(self):
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
            output = output.decode('utf-8').strip()
            lines = output.split('\n')
            gpus = []
            for line in lines:
                values = line.split(', ')
                gpu = {
                    'utilization': float(values[0]),
                    'memory_used': float(values[1]),
                    'memory_total': float(values[2])
                }
                gpus.append(gpu)
            return gpus
        except:
            return None

    def write_to_log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def monitor_resources(self):
        while not self.stop_event.is_set():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            gpu_info = self.get_gpu_info()

            log_message = f"[{timestamp}] CPU: {cpu_percent}%, RAM: {memory.percent}%"

            if gpu_info:
                for i, gpu in enumerate(gpu_info):
                    log_message += f", GPU{i}: {gpu['utilization']}% (Memory: {gpu['memory_used']}/{gpu['memory_total']} MB)"
            else:
                log_message += ", No GPU information available"

            self.write_to_log(log_message)

            time.sleep(self.interval)

    def start(self):
        print(f"Starting resource monitoring. Log file: {self.log_file}")
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.start()

    def stop(self):
        print("Stopping resource monitoring.")
        self.stop_event.set()
        self.monitor_thread.join()
        print("Resource monitoring stopped.")