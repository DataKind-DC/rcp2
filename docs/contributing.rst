Contributing to ``src``
=======================

The ``src`` package organizes rcp2 project code so that it is easy to document,
test, automate, and reuse in new contexts. The package allows volunteers to
build on each other's work, and it allows Red Cross partners to replicate
results with confidence. We welcome your contributions!

This section provides guidelines for contributing to the package. The guidelines
are not hard rules, but they can help volunteers collaborate and develop a more
maintainable code base. Volunteer contributions make us thrive, and we want to
leverage each other's work to the fullest possible extent.


Package Organization
--------------------

Add details about package organization as they become available.


Workflow
--------

Oftentimes volunteer contributions start as notebooks and transition into
package resources. Jupyter notebooks are great for exploration, prototyping, and
idea sharing, but they are difficult to maintain as project infrastructure.
Consider transitioning your work to ``src`` when:

- The code would benefit from high-quality documentation.
- The code is complex and difficult to test informally.
- Multiple notebooks require the same code.

Code doesn't have to be "final" to be part of ``src``. Indeed, we expect the
package to change as project infrastructure needs evolve.


Documentation
-------------

We should document package code thoroughly so that it is easy for others to use.
Modules and all of their components should include docstrings that describe the
purpose of each unit in plain language. The goal is for teammates to be able to
use each other's code confidently without analyzing the inner mechanics.

For a simple function, a one-liner description and a list of arguments and
return values may be enough. More complex docstrings might include multiple
paragraphs and examples. More detail is usually nice for the reader.

Docstrings can include ReStructuredText_ markup for text formatting, lists,
hyperlinks, etc. The markup renders nicely in html documentation.

We recommend following `Google docstring conventions`_ for uniform
docstrings. The project documentation uses the napoleon_ Sphinx extension to
render these docstrings into readable html.

.. _ReStructuredText_: https://docutils.sourceforge.io/rst.html
.. _Google docstring conventions: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html


Unit Tests
----------

We should strive to cover all non-trivial package functions with unit tests.
Unit tests ensure that individual functions in ``src`` are working correctly.
They enable us to verify assumptions about the code we write and to make sure
that new code doesn't break old code.

This project uses the pytest_ unit testing framework. New unit tests go in the
```src/tests/`` directory in python scripts named ``test_*.py``. Each test is a
function that starts with the prefix "test\_". To run unit tests, run ::

  $ pytest

in the unit test directory or in one of its parent directories.

When outlining code, it often helps to write a failing unit test first and then
write code to pass the test. This practice can help clarify a coder's thinking
and also helps them stay caught up on test coverage. Unit tests can be tricky,
and it's not always possible to consider all of the relevant test cases ahead of
time. Remember that having some imperfect tests is better than having no tests,
and that unit tests are works in progress like everything else.

.. _pytest: https://docs.pytest.org/en/stable/


Coding Style
------------

In general, we recommend following `PEP 8`_ style conventions.

Some other suggestions:

- Aim for modular code. Each function has one and only one job.
- Don't repeat yourself. Factor out code instead of copy-pasting.
- Keep lines to 80 characters max to prevent horizontal scrolling.

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/


Successful Pull Requests
------------------------

Some tips to make it easy to merge new pull requests:

- Make sure unit tests pass before submitting.
- Resolve conflicts with the DataKind master branch before submitting.
- Include documentation and unit tests with the pull request.
- Make bite-sized pull requests that are easy to review.
- Link pull requests to existing issues.
- Document the pull requests well with notes and questions.
