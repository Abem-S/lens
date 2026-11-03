.PHONY: install dev dev-https serve test build eval deploy-gpu deploy-glimmer deploy-web test-modal eval-modal

install:            ## install backend + frontend dependencies
	cd backend && uv sync --extra eval --extra gpu
	cd frontend && npm install

dev:                ## backend on :8000 + frontend on :5173 (open http://localhost:5173)
	@trap 'kill 0' INT TERM EXIT; \
	(cd backend && uv run --extra gpu uvicorn lens.app:create_app --factory --reload --host 0.0.0.0 --port 8000) & \
	(cd frontend && npm run dev) & \
	wait

dev-https:          ## same, with self-signed HTTPS so a phone on the LAN can use its camera
	@trap 'kill 0' INT TERM EXIT; \
	(cd backend && uv run --extra gpu uvicorn lens.app:create_app --factory --reload --host 0.0.0.0 --port 8000) & \
	(cd frontend && LENS_HTTPS=1 npm run dev) & \
	wait

build:              ## build the frontend into frontend/dist (served by the backend)
	cd frontend && npm run build

serve: build        ## one process: backend serves the built app on :8000
	cd backend && uv run --extra gpu uvicorn lens.app:create_app --factory --host 0.0.0.0 --port 8000

test:               ## all offline tests
	cd backend && uv run pytest -q
	cd frontend && npm test

eval:               ## A5 scorecard (needs BASETEN; FERMAT also needs HF_TOKEN)
	cd backend && uv run --extra eval lens-eval --source all --limit 20

deploy-gpu:         ## deploy SAM + voice services to Modal (needs MODAL_TOKEN_*, HF_TOKEN)
	$(MODAL) deploy modal_apps/sam.py
	$(MODAL) deploy modal_apps/voice.py
	$(MODAL) deploy modal_apps/sam31.py

# Modal commands read the keys from .env (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET).
MODAL = cd backend && set -a && . ../.env && set +a && uv run --extra gpu modal

deploy-web: build   ## the app on Modal with HTTPS (phone pairing from any network); prints the URL
	$(MODAL) deploy modal_apps/web.py

test-modal:         ## backend tests in a Modal container instead of this machine
	$(MODAL) run modal_apps/ci.py --task test

eval-modal:         ## A5 scorecard in a Modal container; pass ARGS="--source fermat --limit 40"
	$(MODAL) run modal_apps/ci.py --task eval $(if $(ARGS),--args "$(ARGS)",)

deploy-glimmer:     ## optional: Muse Glimmer on Modal (needs GLIMMER_API_KEY in .env)
	$(MODAL) run modal_apps/glimmer.py::download
	$(MODAL) deploy modal_apps/glimmer.py
