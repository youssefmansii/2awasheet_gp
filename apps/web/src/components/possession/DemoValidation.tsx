import { FlaskConical } from 'lucide-react';
import { label, percent, type Manifest } from './api';

const METHODS: Array<[string, string, string]> = [
  ['touch_candidate', 'touch_actor_proxy', 'A nearby player and a change in ball acceleration, or a stable slow arrival.'],
  ['kick_candidate', 'kick_actor_proxy', 'A likely touch with an acceleration peak and reported ball height at or below 0.5 m.'],
  ['completed_pass_candidate', 'completed_pass', 'Interaction transfers between teammates after at least 2 m of ball travel.'],
  ['unsuccessful_pass_candidate', 'unsuccessful_pass', 'An opponent interacts after the ball travels away from the previous player.'],
  ['reception_candidate', '', 'A teammate starts interacting after a predicted completed pass.'],
  ['carry_candidate', 'carry', 'At least 3 m of movement over one second, with likely control through most of the interval.'],
  ['contest_candidate', 'contest', 'Opponents are both near the ball with similar distances. Participants and the subsequent winner are tracked.'],
  ['tackle_candidate', '', 'An opponent gains interaction following a nearby contest; foot contact is not established.'],
  ['interception_candidate', '', 'An opponent gains interaction after a travelling ball without a detected nearby contest.'],
  ['recovery_candidate', '', 'An opponent gains interaction without a clear longer flight.'],
  ['challenge_retained_candidate', '', 'The prior player resumes interaction following a contest.'],
  ['challenge_turnover_candidate', '', 'A different player interacts after the contest.'],
  ['dribble_candidate', '', 'The same player retains interaction after a moving contest.'],
  ['aerial_duel_candidate', '', 'A contest includes a reported ball height above 1.5 m.'],
  ['pressure_candidate', '', 'An opponent stays within 3 m of the interacting player for at least 0.2 seconds.'],
  ['shot_candidate', 'shot', 'A fast release has a projected path towards the opposing goalmouth.'],
  ['cross_candidate', 'cross', 'A completed transfer moves from a wide location into the central attacking area.'],
  ['clearance_candidate', 'clearance', 'A fast release travels away from the defending goal while an opponent is nearby.'],
  ['out_candidate', 'out', 'The tracked ball centre crosses a pitch boundary; this does not prove whole-ball crossing.'],
  ['kickoff_candidate', '', 'A stationary ball is released close to the centre spot.'],
  ['throw_in_candidate', '', 'A stationary ball is released near a touchline.'],
  ['corner_candidate', '', 'A stationary ball is released near a corner.'],
  ['goal_kick_candidate', '', 'A stationary ball is released near the goal area.'],
  ['stationary_restart_candidate', '', 'The ball is released after remaining approximately stationary.'],
  ['goal_line_crossing_candidate', '', 'Reported ball coordinates cross between the posts below the crossbar; no official goal is inferred.']
];

export function DemoValidation({ manifest }: { manifest: Manifest }) {
  const exact = manifest.validation?.frame_actor_proxy?.all;
  const winner = manifest.validation?.event_detection.contest?.winner_among_matched_contests;
  return <div className="pd-validation">
    <div className="pd-explanation"><FlaskConical size={22} aria-hidden="true" /><div>
      <h3>{manifest.validation ? 'Measured against the supplied events' : 'Estimates are ready. Validation needs an events file.'}</h3>
      <p>{manifest.validation ? 'Precision is the share of predicted events that matched. Recall is the share of recorded events found. These results describe this recording only.' : 'Choose an optional events JSON above and analyse again to measure predictions. No accuracy percentages are inferred from the tracking alone.'}</p>
    </div></div>
    <div className="pd-method-grid">
      <section><span className="pd-eyebrow">PLAYER CONTROL</span><h3>A heuristic, not a trained judgement</h3>
        <p>A nearby player, a sustained interaction and compatible relative movement support “likely control”. The learned classifier still needs reviewed labels.</p></section>
      <section><span className="pd-eyebrow">POSSESSION</span><h3>Assigned time, with unknowns visible</h3>
        <p>Team time follows the interacting player and can continue provisionally during ball travel. Contests and missing tracking stay unassigned. This is not official match possession.</p></section>
      <section><span className="pd-eyebrow">EXACT-TIME INTERACTION</span><h3>{exact ? `${percent(exact.player_accuracy_when_covered)} agreement when assigned` : 'Not measured'}</h3>
        <p>{exact ? `${exact.correct_player.toLocaleString()} correct out of ${exact.covered.toLocaleString()} assignments. Coverage is ${percent(exact.carrier_coverage)} of ${exact.labels.toLocaleString()} checked event moments. This differs from touch recovery within a time window.` : 'An events file is required for this check. Interaction, touching the ball and sustained control are separate targets.'}</p></section>
    </div>
    {winner && <p className="pd-winner-note">Contest winner agreement: <strong>{percent(winner.accuracy_when_assigned)}</strong> when assigned ({winner.correct}/{winner.assigned}). This covers only {winner.labels} matched contests, not all tackles.</p>}
    <div className="pd-table-wrap"><table className="pd-table pd-method-table"><thead><tr><th>Event / calculation</th><th>Candidates</th><th>Matched / references</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
      {METHODS.map(([kind, metric, rule]) => {
        const result = manifest.validation?.event_detection[metric];
        return <tr key={kind}><td><strong className="pd-capitalize">{label(kind)}</strong><p>{rule}</p></td>
          <td>{(manifest.candidate_counts[kind] ?? 0).toLocaleString()}</td>
          <td>{result ? `${result.matched} / ${result.references}` : 'Not measured'}</td>
          <td>{result ? percent(result.precision) : '—'}</td><td>{result ? percent(result.reference_recall) : '—'}</td></tr>;
      })}
    </tbody></table></div>
    <p className="pd-note">A dash means the measure is unavailable. Touch/kick and carry references are sparse, so their precision cannot be established. Tackle/interception and restart subtypes remain unvalidated. Thresholds are research-inspired defaults; processing uses later frames and is retrospective.</p>
  </div>;
}
