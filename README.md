# Nexus Sky Engine

The nexus sky engine includes mathematical routines for proactively identifying satellite for observatories, and allows for mitigating and minimizing this inferference.

Basic logic is found in files within src/controllers. Detailed logic is contained within src/utils. Feel free to browse through these files if you're curious how the sky engine functions internally.

The basic idea is as follows:

Given (a) An observatory location (b) An observing field (c) A time horizon, we query leolabs data---and using the returned state vectors---we propagate them across the time horizon and converge towards exactly when and where satellites will interfere with observations.
