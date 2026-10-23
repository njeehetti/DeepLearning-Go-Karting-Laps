// Every circuit shipped in data/tracks/*.json, LYL first (the project's home track), then by name.
import type { TrackDef } from './types';

const files = import.meta.glob<TrackDef>('../../data/tracks/*.json', { eager: true, import: 'default' });

export const TRACKS: readonly TrackDef[] = Object.values(files).sort((a, b) =>
  a.id === 'lyl' ? -1 : b.id === 'lyl' ? 1 : (a.short ?? a.name).localeCompare(b.short ?? b.name, 'fr'),
);

export function trackById(id: string | null | undefined): TrackDef {
  return TRACKS.find((t) => t.id === id) ?? TRACKS[0];
}
