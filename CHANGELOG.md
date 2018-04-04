# Change Log
All notable changes to this project will be documented in this file.

## Unreleased
### Added
- Unit tests for the **datasets** and **networks** module were added.
- The DEFCoN model from the manuscript is located at **defcon/resources/defcon_tf13.h5**. It
  requires Python 3.6, TensorFlow 1.3, and Keras 2.0.8.

### Changed
- Modules were separated into a public and private API. Private
  modules have names that begin with underscores; public modules do
  not.
- All source code was moved from the leb.defcon package to the defcon
  package.

## [v0.0.0]
- Initial release.

[v0.0.0]: https://github.com/LEB-EPFL/DEFCoN/releases/tag/v0.0.0
