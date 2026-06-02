#!/bin/sh

# Step 1: CMYK conversion, preserve image quality
gs -dBATCH -dNOPAUSE \
   -sDEVICE=pdfwrite \
   -sColorConversionStrategy=CMYK \
   -dProcessColorModel=/DeviceCMYK \
   -sDefaultCMYKProfile="$(find ~ -name 'ISOcoated_v2_eci.icc' 2>/dev/null | head -1)" \
   -dDownsampleColorImages=false \
   -dDownsampleGrayImages=false \
   -dDownsampleMonoImages=false \
   -o out_cmyk.pdf main.pdf

# Step 2: Scale to A1
gs -dBATCH -dNOPAUSE \
   -sDEVICE=pdfwrite \
   -dFIXEDMEDIA \
   -dDEVICEWIDTHPOINTS=1684 \
   -dDEVICEHEIGHTPOINTS=2384 \
   -dPDFFitPage \
   -dDownsampleColorImages=false \
   -dDownsampleGrayImages=false \
   -dDownsampleMonoImages=false \
   -o out_cmyk_A1.pdf out_cmyk.pdf