/*
    This class taken from the UCR Suite. The original license is below.
 */
    
/***********************************************************************/
/************************* DISCLAIMER **********************************/
/***********************************************************************/
/** This UCR Suite software is copyright protected © 2012 by          **/
/** Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen,            **/
/** Gustavo Batista and Eamonn Keogh.                                 **/
/**                                                                   **/
/** Unless stated otherwise, all software is provided free of charge. **/
/** As well, all software is provided on an "as is" basis without     **/
/** warranty of any kind, express or implied. Under no circumstances  **/
/** and under no legal theory, whether in tort, contract,or otherwise,**/
/** shall Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen,      **/
/** Gustavo Batista, or Eamonn Keogh be liable to you or to any other **/
/** person for any indirect, special, incidental, or consequential    **/
/** damages of any character including, without limitation, damages   **/
/** for loss of goodwill, work stoppage, computer failure or          **/
/** malfunction, or for any and all other damages or losses.          **/
/**                                                                   **/
/** If you do not agree with these terms, then you you are advised to **/
/** not use this software.                                            **/
/***********************************************************************/
/***********************************************************************/

#include <stdlib.h>

#include "deque.h"

/// Initial the queue at the begining step of envelop calculation
void deq_new(deque *d, length_t capacity)
{
    d->capacity = capacity;
    d->size = 0;
    d->dq = (deq_data *) malloc(sizeof(deq_data)*d->capacity);
    d->f = 0;
    d->r = d->capacity-1;
}

/// Destroy the queue
void deq_free(deque *d)
{
    free(d->dq);
}

/// Insert to the queue at the back
void deq_push_back(deque *d, deq_data v)
{
    d->dq[d->r] = v;
    d->r--;
    if (d->r < 0)
        d->r = d->capacity - 1;
    d->size++;
}

/// Delete the current (front) element from queue
void deq_pop_front(deque *d)
{
    d->f--;
    if (d->f < 0)
        d->f = d->capacity - 1;
    d->size--;
}

/// Delete the last element from queue
void deq_pop_back(deque *d)
{
    d->r = (d->r + 1) % d->capacity;
    d->size--;
}

/// Get the value at the current position of the circular queue
deq_data deq_front(deque *d)
{
    idx_t aux = d->f - 1;

    if (aux < 0)
        aux = d->capacity - 1;
    return d->dq[aux];
}

/// Get the value at the last position of the circular queueint back(deque *d)
deq_data deq_back(deque *d)
{
    idx_t aux = (d->r + 1) % d->capacity;
    return d->dq[aux];
}

/// Check whether or not the queue is empty
short int deq_empty(deque *d)
{
    return d->size == 0;
}

// clear the data in the deque
void deq_clear(deque *d) {
	while (!deq_empty(d)) {
		deq_pop_back(d);
	}
}
