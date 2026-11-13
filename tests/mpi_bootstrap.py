import os
import time
import socket


def _host_ip():
    peer = (os.environ.get('PADDLE_TRAINERS', '').split(',')[0] or '').strip() or '8.8.8.8'
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((peer, 80))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def setup_paddle_env(base_port: int = 36100, rv_dir: str = None):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    world = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    local_size = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

    rv_dir = rv_dir or os.environ.get('EP_RV_DIR')
    assert rv_dir, 'need rendezvous dir (EP_RV_DIR) on shared fs'
    os.makedirs(rv_dir, exist_ok=True)

    ip = _host_ip()
    my_port = base_port + local_rank
    with open(os.path.join(rv_dir, f'{rank}.tmp'), 'w') as f:
        f.write(f'{ip}:{my_port}')
    os.replace(os.path.join(rv_dir, f'{rank}.tmp'), os.path.join(rv_dir, f'{rank}'))

    deadline = time.time() + 120
    eps = [None] * world
    while time.time() < deadline:
        missing = False
        for r in range(world):
            p = os.path.join(rv_dir, str(r))
            if not os.path.exists(p):
                missing = True
                break
            if eps[r] is None:
                try:
                    eps[r] = open(p).read().strip()
                except Exception:
                    missing = True
                    break
        if not missing and all(eps):
            break
        time.sleep(0.3)
    assert all(eps), f'rendezvous timeout, got {eps}'

    os.environ['PADDLE_TRAINER_ID'] = str(rank)
    os.environ['PADDLE_TRAINERS_NUM'] = str(world)
    os.environ['PADDLE_CURRENT_ENDPOINT'] = eps[rank]
    os.environ['PADDLE_TRAINER_ENDPOINTS'] = ','.join(eps)
    os.environ['PADDLE_LOCAL_RANK'] = str(local_rank)
    os.environ['FLAGS_selected_gpus'] = str(local_rank)
    os.environ.setdefault('WORLD_SIZE', '1') 
    return rank, world, local_rank, local_size, eps
