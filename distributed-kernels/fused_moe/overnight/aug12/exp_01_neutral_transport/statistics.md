# exp_01 statistical plan

Frozen before Stage-0 timing.

## Inference unit and summaries

- A campaign is the independent unit. Iterations within a campaign estimate
  that campaign's p50/p95 and are not treated as independent replicates.
- The primary point estimate is the paired median difference in synchronized
  global p50 makespan, with method order rotated within each campaign.
- Also report the paired log ratio, p95 makespan, rank tail, and explanatory
  GB/s. Method selection uses p50 makespan.
- Confidence intervals use a 10,000-resample paired cluster bootstrap over
  campaigns. No rank- or iteration-level bootstrap is admissible.

## Discovery and confirmation

- Discovery sweep: three paired campaigns per method/size stratum, sufficient
  only to locate candidate crossovers or effects above the pre-registered 2%
  expansion threshold.
- Confirmation: fresh campaigns after the crossover points are selected.
  Run at least five paired campaigns for effects at or above 2%.
- A difference below 2% requires an explicit power calculation and at least
  seven paired campaigns. Continue until 80% power for the observed paired
  campaign variance or 15 campaigns, then report underpowered if unresolved.
- Sizes selected from discovery are confirmed one point below, nearest, and
  one point above each crossover. Discovery campaigns do not enter the final
  confidence interval.

## Margins and decisions

- Practical-equivalence margin: ±2% of the paired baseline p50.
- Method/regime interaction expansion threshold: 3%.
- Superiority: paired 95% CI excludes zero and the point estimate improves
  makespan by at least 2%.
- Equivalence: the paired 90% CI lies wholly inside ±2%.
- Otherwise: unresolved at the achieved power.
- A method more than 10% slower in every Stage-2 corner is not expanded unless
  its pre-registered skew, topology, direction, or SDMA rescue regime remains
  untested.

## Multiplicity

- Within each record-size family, Holm-adjust the superiority p-values for all
  methods compared with the pre-registered CU-push reference.
- Push-versus-pull and same-API MORI P2P-versus-SDMA are separate named
  contrasts and each receives its own Holm family over record sizes.
- Confirmation points selected by discovery are explicitly selective; their
  CIs are reported as confirmation-only estimates from independent campaigns.

## Crossover definition

A crossover exists only when:

1. adjacent tested sizes give opposite signs for the paired makespan contrast,
2. both methods pass identical correctness/executor strata, and
3. independent confirmation supports the sign on at least one side.

Interpolation is log-linear in record bytes and is reported as an interval
between tested sizes, never as an exactly measured byte count.
