all: requirements-gpu-cuda13.txt requirements-gpu-cuda12.txt requirements-gpu-cuda13_local.txt requirements-gpu-cuda12_local.txt requirements-local-testing.txt


requirements-gpu-cuda13.txt: uv.lock
	uv pip compile pyproject.toml --extra gpu_cuda13 -o requirements-gpu-cuda13.txt
requirements-gpu-cuda12.txt: uv.lock
	uv pip compile pyproject.toml --extra gpu_cuda13 -o requirements-gpu-cuda12.txt
requirements-gpu-cuda13_local.txt: uv.lock
	uv pip compile pyproject.toml --extra gpu_cuda13_local -o requirements-gpu-cuda13_local.txt
requirements-gpu-cuda12_local.txt: uv.lock
	uv pip compile pyproject.toml --extra gpu_cuda12_local -o requirements-gpu-cuda12_local.txt
requirements-local-testing.txt: uv.lock
	uv pip compile pyproject.toml --extra personal -o requirements-local-testing.txt
