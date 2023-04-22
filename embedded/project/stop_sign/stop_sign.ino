#include <Arduino.h>
#include <TensorFlowLite.h>

#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 200 * 1024; //min value that works is twice the size of every tensor, ie need input and output present, ie pick biggiest layer and double it
//keep increasing until linker error, ie claims all of the 1meg and itll fail
static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


void setup() {
  delay(5000);  // Delay for 5 seconds

  Serial.println("Proj 1 waking up");  // Print startup message
  
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model provided is schema version %d not equal to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

static tflite::MicroMutableOpResolver<10> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddRelu();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();  


  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;


  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

    // Print the shape of the input tensor.
  Serial.print("Input tensor shape: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("]");


  // Check if input tensor is not null
  if (input == nullptr) {
    error_reporter->Report("Input tensor not initialized");
    return;
  }

  // Check if input dimensions match expected values
  if (input->dims->size != 4 || input->dims->data[0] != 1 || input->dims->data[1] != kNumRows ||
      input->dims->data[2] != kNumCols || input->dims->data[3] != kNumChannels) {
    error_reporter->Report("Input tensor has incorrect dimensions");
    return;
  }
  
  Serial.println("Setup complete");  // Print setup complete message
}


void loop() {
Serial.println("Starting Loop");  // Print setup complete message
unsigned long t1 = micros();
// Get image from provider.
if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input->data.int8)) {
  TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  return;
}

Serial.println("Got image");  // Print setup complete message

Serial.print("Input tensor shape: [");
Serial.print(input->dims->data[0]);
Serial.print(", ");
Serial.print(input->dims->data[1]);
Serial.print(", ");
Serial.print(input->dims->data[2]);
Serial.print(", ");
Serial.print(input->dims->data[3]);
Serial.println("]");

Serial.println("Starting invoke");  // Print setup complete message

// Run the model on this input and make sure it succeeds.
TfLiteStatus invoke_status = interpreter->Invoke();

Serial.println("invoke status set");  // Print setup complete message

if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke failed with status %d", invoke_status);
  return;
}

Serial.println("Pass invoke");  // Print setup complete message

TfLiteTensor* output = interpreter->output(0);
Serial.print("Output tensor shape: [");
for (int i = 0; i < output->dims->size; i++) {
  Serial.print(output->dims->data[i]);
  if (i < output->dims->size - 1) {
    Serial.print(", ");
  }
}
Serial.println("]");

// Process the inference results.
int8_t person_score = output->data.uint8[kPersonIndex];
int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
RespondToDetection(error_reporter, person_score, no_person_score);
Serial.println("Getting scores");  // Print setup complete message

// Measure the time for a single inference
unsigned long t2 = micros();

// Print out the measured times
Serial.print("Inference time: ");
Serial.print(t2 - t1);
Serial.println(" microseconds");

}

