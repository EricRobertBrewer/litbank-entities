#! /bin/bash

for CATEGORY in PER FAC GPE LOC VEH ORG
do
  for ((i = 1; i <= 10; i++))
  do
    python -m litbank_entities.extract bert --folds 10 --fold $i --categories $CATEGORY
  done
done
