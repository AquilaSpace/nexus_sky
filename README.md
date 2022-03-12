# Nexus Sky Engine

The nexus sky engine includes mathematical routines for determining when satellites will interfere with astronomy observations, and contains functionality to mitigate and minimize this inferference.

Most logic is found in files within src/controllers. Feel free to browse through them if you're curious how the sky engine functions internally.

The basic idea is as follows:

Given (a) An observatory location (b) An observing field (c) A time horizon, we query leolabs data---and using the returned state vectors---we zero in on exactly when and where satellites will interfere with observations.
