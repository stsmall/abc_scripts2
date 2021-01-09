# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:18:46 2020
@author: Scott T. Small

Reading and parsing ms-style formatted files for input into abc_stats.py

"""
import bisect


def getSnpsOverflowingChr(newPositions, totalPhysLen):
    """

    Parameters
    ----------
    newPositions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.

    Returns
    -------
    overflowers : TYPE
        DESCRIPTION.

    """
    overflowers = []
    for i in reversed(range(len(newPositions))):
        if newPositions[i] > totalPhysLen:
            overflowers.append(newPositions[i])
    return overflowers


def fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers):
    """

    Parameters
    ----------
    newPositions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.
    overflowers : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    posH = {}
    for pos in newPositions:
        posH[pos] = 1
    for i in range(len(overflowers)):
        del newPositions[-1]
    for pos in reversed(range(1, totalPhysLen+1)):
        if pos not in posH:
            bisect.insort_left(newPositions, pos)
            overflowers.pop()
            if len(overflowers) == 0:
                break


def discrete_positions(positions, totalPhysLen):
    """

    Parameters
    ----------
    positions : TYPE
        DESCRIPTION.
    totalPhysLen : TYPE
        DESCRIPTION.

    Returns
    -------
    newPositions : TYPE
        DESCRIPTION.

    """
    snpNum = 1
    prevPos = -1
    prevIntPos = -1
    newPositions = []
    for position in positions:
        assert position >= 0 and position <= 1., "Mutations positions must all be in [0, 1)"
        assert position >= prevPos
        origPos = position
        if position == prevPos:
            position += 0.000001
        prevPos = origPos

        intPos = int(totalPhysLen*position)
        if intPos == 0:
            intPos = 1
        if intPos <= prevIntPos:
            intPos = prevIntPos + 1
        prevIntPos = intPos
        newPositions.append(intPos)
    overflowers = getSnpsOverflowingChr(newPositions, totalPhysLen)
    if overflowers:
        fillInSnpSlotsWithOverflowers(newPositions, totalPhysLen, overflowers)
    assert len(newPositions) == len(positions)
    assert all(newPositions[i] <= newPositions[i+1]
               for i in range(len(newPositions)-1))
    assert newPositions[-1] <= totalPhysLen
    return newPositions
