import JSZip from "jszip";
import fileSaver from "file-saver";
import type { GeneratedFile } from "@/lib/types";

export async function downloadZip(files: GeneratedFile[], name = "generated-project.zip") {
  if (!files.length) return;

  const zip = new JSZip();
  for (const file of files) {
    zip.file(file.path, file.content);
  }

  const blob = await zip.generateAsync({ type: "blob" });
  const { saveAs } = fileSaver;
  saveAs(blob, name);
}
