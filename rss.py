import math

from control import StateSpace
from control.iosys import _process_iosys_keywords
from control.iosys import _process_signal_list
from numpy import any
from numpy import cos
from numpy import exp
from numpy import sin
from numpy import zeros
from numpy.linalg import LinAlgError
from numpy.linalg import solve


def _rand(np_random, *args):
    return np_random.uniform(low=0, high=1, size=args)


def _randn(np_random, *args):
    return np_random.standard_normal(size=args)


def rss(np_random, states=1, outputs=1, inputs=1, strictly_proper=False, **kwargs):
    """Create a stable random state space object.

    Parameters
    ----------
    states, outputs, inputs : int, list of str, or None
        Description of the system states, outputs, and inputs. This can be
        given as an integer count or as a list of strings that name the
        individual signals.  If an integer count is specified, the names of
        the signal will be of the form 's[i]' (where 's' is one of 'x',
        'y', or 'u').
    strictly_proper : bool, optional
        If set to 'True', returns a proper system (no direct term).
    dt : None, True or float, optional
        System timebase. 0 (default) indicates continuous
        time, True indicates discrete time with unspecified sampling
        time, positive number is discrete time with specified
        sampling time, None indicates unspecified timebase (either
        continuous or discrete time).
    name : string, optional
        System name (used for specifying signals). If unspecified, a generic
        name <sys[id]> is generated with a unique integer id.

    Returns
    -------
    sys : StateSpace
        The randomly created linear system.

    Raises
    ------
    ValueError
        if any input is not a positive integer.

    Notes
    -----
    If the number of states, inputs, or outputs is not specified, then the
    missing numbers are assumed to be 1.  If dt is not specified or is given
    as 0 or None, the poles of the returned system will always have a
    negative real part.  If dt is True or a postive float, the poles of the
    returned system will have magnitude less than 1.

    """
    # Process keyword arguments
    kwargs.update({"states": states, "outputs": outputs, "inputs": inputs})
    name, inputs, outputs, states, dt = _process_iosys_keywords(kwargs)

    # Figure out the size of the sytem
    nstates, _ = _process_signal_list(states)
    ninputs, _ = _process_signal_list(inputs)
    noutputs, _ = _process_signal_list(outputs)

    sys = _rss_generate(
        np_random,
        nstates,
        ninputs,
        noutputs,
        "c" if not dt else "d",
        name=name,
        strictly_proper=strictly_proper,
    )

    return StateSpace(
        sys, name=name, states=states, inputs=inputs, outputs=outputs, dt=dt, **kwargs
    )


def _rss_generate(
    np_random, states, inputs, outputs, cdtype, strictly_proper=False, name=None
):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    rand = lambda *args: _rand(np_random, *args)
    randn = lambda *args: _randn(np_random, *args)

    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." % states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." % inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." % outputs)
    if cdtype not in ["c", "d"]:
        raise ValueError("cdtype must be `c` or `d`")

    # Make some poles for A.  Preallocate a complex array.
    poles = zeros(states) + zeros(states) * 0.0j
    i = 0

    while i < states:
        if rand() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i : i + 2] = poles[i - 2 : i]  # noqa: E203
                i += 2
        elif rand() < pReal or i == states - 1:
            # No-oscillation pole.
            if cdtype == "c":
                poles[i] = -exp(randn()) + 0.0j
            else:
                poles[i] = 2.0 * rand() - 1.0
            i += 1
        else:
            # Complex conjugate pair of oscillating poles.
            if cdtype == "c":
                poles[i] = complex(-exp(randn()), 3.0 * exp(randn()))
            else:
                mag = rand()
                phase = 2.0 * math.pi * rand()
                poles[i] = complex(mag * cos(phase), mag * sin(phase))
            poles[i + 1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = randn(states, states)
        try:
            A = solve(T, A) @ T  # A = T \ A @ T
            break
        except LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = randn(states, inputs)
    C = randn(outputs, states)
    D = randn(outputs, inputs)

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rand(states, inputs) < pBCmask
        if any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rand(outputs, states) < pBCmask
        if any(Cmask):  # Retry if we get all zeros.
            break
    if rand() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rand(outputs, inputs) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else zeros(D.shape)

    if cdtype == "c":
        ss_args = (A, B, C, D)
    else:
        ss_args = (A, B, C, D, True)
    return StateSpace(*ss_args, name=name)
