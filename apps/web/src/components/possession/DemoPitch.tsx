import type { DemoFrame } from './api';

export function DemoPitch({ frame, selected, onSelect, labels }: {
  frame: DemoFrame | null; selected: string; onSelect: (key: string) => void; labels: boolean;
}) {
  return <svg className="pd-pitch" viewBox="0 0 120 82" role="img" aria-label="Football tracking pitch">
    <defs>
      <pattern id="pd-grass" width="15" height="82" patternUnits="userSpaceOnUse">
        <rect width="7.5" height="82" fill="#12271e" /><rect x="7.5" width="7.5" height="82" fill="#142c21" />
      </pattern>
      <radialGradient id="pd-light"><stop stopColor="#4b8054" stopOpacity=".13" /><stop offset="1" stopColor="#0a1710" stopOpacity=".1" /></radialGradient>
    </defs>
    <rect width="120" height="82" rx="2" fill="#0c1c15" />
    <rect x="7.5" y="7" width="105" height="68" fill="url(#pd-grass)" />
    <rect width="120" height="82" fill="url(#pd-light)" />
    <g fill="none" stroke="#70927d" strokeWidth=".16" opacity=".7">
      <rect x="7.5" y="7" width="105" height="68" />
      <path d="M60 7v68" /><circle cx="60" cy="41" r="9.15" />
      <rect x="7.5" y="20.84" width="16.5" height="40.32" /><rect x="96" y="20.84" width="16.5" height="40.32" />
      <rect x="7.5" y="31.84" width="5.5" height="18.32" /><rect x="107" y="31.84" width="5.5" height="18.32" />
      <rect x="5.5" y="37.34" width="2" height="7.32" /><rect x="112.5" y="37.34" width="2" height="7.32" />
      <path d="M24 33.7a9.15 9.15 0 0 1 0 14.6M96 33.7a9.15 9.15 0 0 0 0 14.6" />
    </g>
    <g fill="#70927d"><circle cx="60" cy="41" r=".25" /><circle cx="18.5" cy="41" r=".25" /><circle cx="101.5" cy="41" r=".25" /></g>
    <text x="8" y="4.4" className="pd-pitch-caption">HOME</text><text x="112" y="4.4" textAnchor="end" className="pd-pitch-caption">AWAY</text>
    {frame?.players.map((player) => <g key={player.key} transform={`translate(${player.x + 60} ${41 - player.y})`}
      role="button" tabIndex={0} aria-label={`Select ${player.name}, ${player.side} ${player.jersey}`}
      onClick={() => onSelect(player.key)} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(player.key); } }}
      className="pd-pitch-player">
      <title>{player.name} · {player.side} #{player.jersey}</title>
      {player.key === frame.carrier && <circle r="2.5" fill="#c8f45a" fillOpacity=".1" stroke="#c8f45a" strokeWidth=".25" />}
      {player.key === selected && <circle r="3" fill="none" stroke="white" strokeWidth=".18" strokeDasharray=".5 .5" />}
      <circle r="1.45" fill={player.side === 'home' ? '#77baff' : '#efb17c'} stroke="#0d1914" strokeWidth=".25" />
      {labels && <text textAnchor="middle" dominantBaseline="central" fill="#10202a" fontSize="1.45" fontWeight="800">{player.jersey}</text>}
    </g>)}
    {frame?.ball && <g transform={`translate(${frame.ball.x + 60} ${41 - frame.ball.y})`}>
      {frame.ball.reconstructed && <circle r="1.25" fill="none" stroke="white" strokeWidth=".13" strokeDasharray=".3 .3" />}
      <circle r=".62" fill="#fff" stroke="#09130e" strokeWidth=".2" /><title>{frame.ball.reconstructed ? 'Reconstructed ball position' : 'Ball'}</title>
    </g>}
    {!frame && <g textAnchor="middle" fill="#9ab1a2"><text x="60" y="39" fontSize="2.7" fontWeight="600">Your recording, on the pitch.</text>
      <text x="60" y="44" fontSize="1.6">Load a tracking file to explore the game.</text></g>}
    {frame && !frame.ball && <g><rect x="42" y="67" width="36" height="4.5" rx="1" fill="#111d17" />
      <text x="60" y="70" textAnchor="middle" fontSize="1.6" fill="#e4cfa7">Ball position unavailable</text></g>}
  </svg>;
}
