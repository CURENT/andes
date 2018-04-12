#!/usr/bin/env bash

andesplot ieee39_out.dat -d 0 460 461 &
andesplot ieee39_out.dat 0 207 229 458 462 -d &
andesplot ieee39_out.dat -d 0 464 465 &
andesplot ieee39_out.dat -d 0 268 229 207 458 460 461 462 464 &