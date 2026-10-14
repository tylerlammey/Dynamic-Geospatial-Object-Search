/** Bounded tile requests and Leaflet retention. No application state lives here. */
import L from 'leaflet';

type TileJob = {
  img: HTMLImageElement;
  coords: L.Coords;
  done: L.DoneCallback;
  controller: AbortController;
  cancelled: boolean;
  objectUrl?: string;
};

type TileOptions = {
  route: (coords: L.Coords) => string;
  report: (message: string) => void;
};

class TilePool {
  private queue: TileJob[] = [];
  private active = 0;
  private failures = 0;

  constructor(private options: TileOptions) {}

  enqueue(job: TileJob): void {
    this.queue.push(job);
    this.pump();
  }

  pump(): void {
    this.queue = this.queue.filter(job => !job.cancelled);
    while (this.active < 4 && this.queue.length) {
      const job = this.queue.shift()!;
      this.active++;
      void this.fetchTile(job).finally(() => { this.active--; this.pump(); });
    }
    this.options.report(this.active + this.queue.length
      ? `Loading local imagery · ${this.active} active${this.queue.length ? ` / ${this.queue.length} waiting` : ''}`
      : this.failures ? 'Some tiles could not load. Pan or use Fit state to retry.' : '');
  }

  private async fetchTile(job: TileJob): Promise<void> {
    try {
      if (job.cancelled) return;
      const route = this.options.route(job.coords);
      let response = await fetch(route, {signal: job.controller.signal});
      for (let retry = 0; response.status === 503 && retry < 2 && !job.cancelled; retry++) {
        await new Promise(resolve => setTimeout(resolve, 500 * (retry + 1)));
        response = await fetch(route, {signal: job.controller.signal});
      }
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Tile read failed');
      }
      const blob = await response.blob();
      if (job.cancelled) return;
      job.objectUrl = URL.createObjectURL(blob);
      job.img.onload = () => { if (!job.cancelled) job.done(undefined, job.img); };
      job.img.onerror = () => { if (!job.cancelled) job.done(new Error('Tile image could not decode'), job.img); };
      job.img.src = job.objectUrl;
    } catch (error) {
      if (!job.cancelled) { this.failures++; job.done(error as Error, job.img); }
    }
  }
}

export function createLocalTiles(options: TileOptions): L.GridLayer {
  const pool = new TilePool(options);
  const jobs = new WeakMap<HTMLElement, TileJob>();
  const LocalTiles = L.GridLayer.extend({
    createTile(coords: L.Coords, done: L.DoneCallback) {
      const img = document.createElement('img');
      img.alt = '';
      img.setAttribute('role', 'presentation');
      const job = {img, coords, done, controller: new AbortController(), cancelled: false};
      jobs.set(img, job);
      pool.enqueue(job);
      return img;
    },
    _abortLoading(this: L.GridLayer & {_tiles: Record<string, {coords: L.Coords; loaded?: number}>; _tileZoom: number; _removeTile: (key: string) => void}) {
      // Loaded tiles stay available for scaling/fading beneath replacements.
      for (const [key, tile] of Object.entries(this._tiles || {})) {
        if (tile.coords.z !== this._tileZoom && !tile.loaded) this._removeTile(key);
      }
    },
  });
  const TileClass = LocalTiles as typeof L.GridLayer;
  const tiles = new TileClass({tileSize: 256, keepBuffer: 1, updateWhenIdle: true, updateWhenZooming: false, bounds: [[-1, -180], [85, 180]], minZoom: 5, maxZoom: 20});
  tiles.on('tileunload', (event: L.TileEvent) => {
    const job = jobs.get(event.tile);
    if (!job) return;
    job.cancelled = true;
    job.controller.abort();
    if (job.objectUrl) URL.revokeObjectURL(job.objectUrl);
    pool.pump();
  });
  return tiles;
}
