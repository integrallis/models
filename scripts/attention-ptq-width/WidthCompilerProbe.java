package com.integrallis.models.backend.purejava.ops;

/** Keeps Java result messages separate from the VM's assembly output. */
public final class WidthCompilerProbe {
  public static void main(String[] args) {
    System.setOut(System.err);
    WidthRiskProbe.main(args);
  }
}
