# Element semi-apertures are derived, not declared

## Why

The paper never gives element semi-diameters as inputs. A system's aperture is
specified by entrance pupil diameter or f-number (Supp. S2.1, Table S8); how
wide each individual element has to be is a *consequence* of the light the
system carries, not a design variable. Our optics classes follow that: the
clear semi-diameter of every surface is derived from the ray footprint.

This is not a cosmetic choice. During optimization, curvatures and thicknesses
move, and the off-axis beam walks up and down the rear elements. Any
hand-declared semi-diameter goes stale immediately. On the two-element toy,
declaring the apertures that looked right for the starting design and then
running 40 iterations of TRA-LM gives:

| surface | declared | required @ start | required @ optimum | overflow @ optimum |
|---------|---------:|-----------------:|-------------------:|-------------------:|
| 0       | 2.60 mm  | 3.140 mm         | 2.913 mm           | **+0.313 mm**      |
| 1       | 2.80 mm  | 3.128 mm         | 3.141 mm           | **+0.341 mm**      |
| 2       | 3.40 mm  | 2.812 mm         | 4.537 mm           | **+1.137 mm**      |
| 3       | 3.80 mm  | 2.661 mm         | 4.574 mm           | **+0.774 mm**      |

`required` is the largest ray radius reaching that surface over the full
field × wavelength × pupil grid. Every surface ends up carrying light outside
its declared rim, and the rear surface misses by 20 %. In the layout figure this
showed up as rays visibly refracting at a point above the glass they were drawn
to refract at — a physically impossible picture produced by correct ray tracing
plus a wrong aperture annotation.

Derived apertures cannot go stale, because they are recomputed from the same
trace that produced the rays being drawn.

## API

```python
optics.clear_semi_apertures(margin=1.05)   # (S,) tensor, DERIVED from the trace
optics.effective_semi_apertures(margin)    # derived, with declared values overriding
optics.semi_aperture                       # list[float] actually in force (cached)
optics.aperture_report(margin=1.05)        # per-surface declared vs required table
```

`margin` scales the required radius to leave a mechanical rim (5 % by default).
All three return **detached** values: apertures are geometry for drawing and
reporting, never optimization variables. Nothing in the residuals, the Jacobian,
or the task loss depends on them, so there is no gradient to lose.

`semi_aperture` is cached against a cheap fingerprint of the parameter vector,
so repeated layout calls on an unchanged design do not re-trace. Any parameter
change invalidates the cache automatically.

## Declaring an override

Pass a semi-aperture on a `Surface` when the physical element really is
narrower than the light — a stop, a deliberately vignetting baffle, a part you
have already had made:

```python
Surface(c=0.02, thickness=1.5, n_after=1.5168, semi_aperture=3.0)   # override
Surface(c=0.02, thickness=1.5, n_after=1.5168)                      # derived
```

An override is honoured verbatim for drawing. It does **not** clip rays — v1
traces without vignetting (`aim_rays` is a stub, Supp. S1.3.2). So a declared
aperture narrower than the footprint means the figure understates the beam, and
`aperture_report()["overflow"]` tells you by how much. Check it after
optimization:

```python
for row in optics.aperture_report():
    if row["overflow"] > 0:
        print(f"S{row['surface']}: rays exceed declared rim by "
              f"{row['overflow']:.3f} mm; recommend {row['recommended']:.3f} mm")
```

## Interaction with the layout view

Derived apertures exposed a separate bug in the layout drawing. The ray fan was
launched at the smallest clear semi-aperture, which was a safe choice when those
values were declared by hand but not once they are derived: the derived minimum
is 3.30 mm against an entrance-pupil radius of 2.50 mm, so the figure drew light
the aperture stop blocks. At 30 deg those extra rays miss the rear element, and
the layout tracer -- which had no validity mask, unlike the main tracer -- let
Newton converge on the far branch of the conic, producing a vertex 26 mm past a
sensor sitting at 12 mm and a ray doubling back to y = 69 mm.

Both halves are fixed. `_limiting_radius` caps the fan at `epd/2`, and
`ray_paths` now applies the same reachability, back-march and TIR tests as
`_trace_packed`, writing NaN for a ray's remaining vertices so the polyline
stops where the ray does.

`plot_layout` and `compare_designs` still clamp the y-view to the element extent
(`_clamp_layout_view`, keep factor 1.6) and print how far the rays reach inside
the axes, so a clipped frame is never mistaken for the true extent. With the two
fixes above neither toy reaches the clamp -- it remains as a guard for designs
that genuinely diverge. `plot_layout(..., y_view="rays")` disables it.

One matplotlib detail matters here: the cross-section sets an equal aspect with
`adjustable='datalim'`, under which fixed y-limits are silently discarded to
satisfy the aspect (it warns "Ignoring fixed y limits..."). The clamp therefore
switches to `adjustable='box'` before setting limits, otherwise it would
annotate a half-width the panel does not show.
