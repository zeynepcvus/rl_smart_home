# Smart Home RL Refactor Notes

## What changed
- Hourly electricity price is generated once per hour and reused consistently in observation and reward.
- Daily scenarios are explicit and shareable across evaluation runs.
- User behavior is dynamic through occupancy, awake/sleep schedule, and lighting demand.
- Observation space now exposes operationally important state:
  - user awake flag
  - night flag
  - occupancy
  - lighting need
  - hours left in day
  - per-device completion state
  - deadline urgency
  - user priority
  - preferred window signal
- Reward is rebalanced around:
  - cost
  - comfort
  - task reliability
- Evaluation is paired:
  - RL and rule-based agents are tested on identical scenarios
  - extra metrics are reported
- Rule-based baseline is stronger and more realistic than the earlier version.

## Expected benefits
- Cleaner credit assignment for PPO
- Fairer evaluation
- Better alignment with project goal:
  minimize cost while preserving user comfort and task completion
