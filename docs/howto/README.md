# How-to guides

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). This page is an index;
> each guide below carries its own verification stamp.

Task-oriented recipes for someone who has already finished [`docs/tutorial.md`](../tutorial.md).

**These pages cannot promise success — the tutorial can.** The tutorial runs on a committed fixture
chosen to avoid every hard case; these pages run on *your* data, on *your* infrastructure, and each
one names the pitfalls it knows about. When one of them fails on you, that is a bug report worth
filing (`gh issue create`), not something to work around silently.

| you want | read | it assumes |
|---|---|---|
| to point rise-eo at your own ROI and your own labels | [`your-own-region.md`](your-own-region.md) | you finished the tutorial |
| to know what a real download actually costs before you start it | [`download-real-imagery.md`](download-real-imagery.md) | CDSE credentials, or MPC (anonymous) |
| to build the two container images the cluster runs | [`build-the-images.md`](build-the-images.md) | `az login` + a workspace your platform admin provisioned |
| to fan the same calls out onto an Azure ML cluster | [`run-at-scale.md`](run-at-scale.md) | a workspace + cluster your platform admin provisioned, and both images built |
| to package your trained model so rise-eo can run it | [`bundle-your-model.md`](bundle-your-model.md) | a model you already fit |
| to serve the output COGs/STAC on a map viewer | [`serve-xyz.md`](serve-xyz.md) | an inference run that finished |

Reading order, if you're working through them: **your-own-region → download-real-imagery →
bundle-your-model → build-the-images → run-at-scale → serve-xyz**. The first two are about getting *your* data in; the
last three are about getting results out.

For a complete, readable script rather than fragments, see
[`examples/eurocrops_rf.py`](../../examples/eurocrops_rf.py).
