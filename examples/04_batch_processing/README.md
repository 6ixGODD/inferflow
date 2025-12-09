# Dynamic Batch Processing Example

Demonstrates adaptive batching for high-throughput inference with peak shaving.

## Features

- Dynamic batch size adaptation
- Request queue management
- Peak traffic handling
- Real-time metrics
- Latency statistics

## Traffic Simulation

The example simulates realistic traffic patterns:

- **Bursts**: Sudden spikes in requests (peak hours)
- **Valleys**: Low traffic periods
- **Staggered requests**: Realistic timing

## Metrics

- Total requests processed
- Average batch size
- Latency statistics (min, mean, p95, max)
- Throughput (req/s)
- Queue depth

## Usage

```bash
python batch_demo.py
```

## Key Benefits

* Throughput: 5-10x improvement vs. sequential
* Latency: Consistent under load
* Efficiency: Automatic batch size optimization
