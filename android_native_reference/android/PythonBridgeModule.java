package com.bloodgroupapp; // Replace with actual package name

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

public class PythonBridgeModule extends ReactContextBaseJavaModule {

    public PythonBridgeModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return "PythonBridge";
    }

    @ReactMethod
    public void predictOffline(String imagePath, Promise promise) {
        try {
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(getReactApplicationContext()));
            }

            Python py = Python.getInstance();
            PyObject predictModule = py.getModule("predict");

            // predict_image is the function inside predict.py
            PyObject result = predictModule.callAttr("predict_image", imagePath);

            promise.resolve(result.toString());

        } catch (Exception e) {
            promise.reject("PYTHON_ERROR", e.getMessage());
        }
    }
}
