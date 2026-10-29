# Mã băm của các file không đưa lên repo

Dùng để đối chiếu với thư mục làm việc gốc (`C:\Users\dungbt06\Fine-tune`). Dùng sha256 đầy đủ trừ khi ghi khác.

## Benchmark
| File | sha256 | Ghi chú |
|---|---|---|
| `data/bench_v1_3/dev_v1_3.jsonl` | `c6a9a75d5ca512a793a6b03151dc4fe36875fbc43a780b1d5846b730a270cd0e` | 3.847 dòng |
| `data/bench_v1_3/test_v1.3.jsonl` | `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da` | đóng băng, chưa chạy model; trùng từng byte test 1.2 |
| `data/bench_v1_2/test_v1.2.jsonl` | `2bdbd2109a327093bd0ece9aa8aadd14c8b7f2d3c37bd80cc8be4f9408bec5da` | |
| `data/bench_v1_2/dev_v1_2.jsonl` | `ef88be1625c83f97c4a2829e1e0b3e647b082487e2cc92a07e7f0c1230c95a6b` | |
| `data/bench_v1_1/test_v1.1.jsonl` | `fbad818b11b7ca8e23767a7287e9db7dae9a6a767042c04eb79c053a8b811b0c` | |
| `data/bench_v1/test_v1.0.jsonl` | `d92d31f696a37cdf7b5d6f95bd1ab899c878e0b4a042a8c7ae691ae8ab4b0d47` | |

## Dữ liệu train
| File | Dòng | sha256 | Dùng cho |
|---|---|---|---|
| `data/train_v0.jsonl` | 20.000 (theo summary) | `d10c9362d2cbade1b5e19790938f06c3a82011276d1da02491b6703e2e1fb466` | v0.1-m1 |
| `data/val_v0.jsonl` | 1.000 | `4377e710aeaebeae7b05d6c95c52d4ede0744674074ed505bb6851cc7c93a8ca` | val của mọi bản |
| `results/train_v1/components/C.jsonl` | 31.034 | `005ca3d547aa2f9e2f356a8f75eb7f3f17addba1cc4d74e3a7b44a7b39c86e89` | v0.2-c |
| `results/train_2b/train_2b_v3.jsonl` | 31.596 | `ea604a0138c20f42b4beae67cdeb3fe02908c0fa96cd63b104c469cc54a11635` | v0.3-2b |
| `results/train_2b_prime/train_2b_prime.jsonl` (V-e) | 33.096 | `c97bb08daa7efcac70f567bda2e4b6cc612288c5cad9c2bdfe273ca750116cfc` | 2b′, chưa train |

## Adapter (LoRA r16, 69.782.384 byte mỗi file)
| Phiên bản | File | sha256 |
|---|---|---|
| v0.1-m1 | `results/ft_loss/train/Qwen3-1.7B-s1/adapter/adapter_model.safetensors` | `4b3087b0d35a65a4a2191c46eb55c192eab17d01b5a032914a2d770765cbfd40` |
| v0.2-c | `results/train_v1/train/C-Qwen3-1.7B-s1/adapter/adapter_model.safetensors` | `45259cdaee27152558e081b1a0160525d4039af2d8187b10f6698fd826fac412` |
| v0.3-2b (s1) | `results/train_2b/train/2b-Qwen3-1.7B-s1/adapter/adapter_model.safetensors` | `3398ef3f22e07a93f0b7753c2ceea1c3694e327ce22e56d9b7115a08c3a826bb` |
| v0.3-2b (s2) | `results/train_2b/train/2b-Qwen3-1.7B-s2/adapter/adapter_model.safetensors` | `2cfcb8bb5773a9682223a7aba05ab2611fef7249f49ad16bfaafe935cb8f74e7` |

Model nền: `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
