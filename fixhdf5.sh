#!/bin/bash
find . -type f -exec sed -i -e 's^"hdf5/serial/hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5/serial/hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;
