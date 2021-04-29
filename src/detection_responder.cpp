/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "detection_responder.h"
#include <Arduino.h>

// This dummy implementation writes person and no person scores to the error
// console. Real applications will want to take some custom action instead, and
// should implement their own versions of this function.
void RespondToDetection(tflite::ErrorReporter *error_reporter, uint8_t person_score, uint8_t no_person_score)
{

    error_reporter->Report("Person score:%3d No person score:%3d", person_score, no_person_score);
    if (person_score > 0.8 * 256)
    {
        digitalWrite(LED_BUILTIN, person_score > no_person_score);
        delay(100);
        digitalWrite(LED_BUILTIN, false);
    }
}
