import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = process.cwd();
const sourcePath = path.join(
  projectRoot,
  "验收",
  "Acceptance 1",
  "子任务6_时间序列特征提取结果.csv",
);
const outputPath = path.join(
  projectRoot,
  "验收",
  "Acceptance 1",
  "子任务6_时间序列特征提取结果_颜色标注.xlsx",
);

const csvText = (await fs.readFile(sourcePath, "utf8")).replace(/^\uFEFF/, "");
const lines = csvText.trimEnd().split(/\r?\n/);
const headers = lines[0].split(",");
const values = lines.map((line, rowIndex) => line.split(",").map((value, colIndex) => {
  if (rowIndex === 0 || value === "" || colIndex === 13) return value;
  if (colIndex === 0) return new Date(`${value}T00:00:00`);
  const number = Number(value);
  return Number.isFinite(number) ? number : value;
}));

const lagColumns = headers.flatMap((header, index) => header.includes("滞后") ? [index] : []);
const windowColumns = headers.flatMap((header, index) => header.includes("滚动") ? [index] : []);
if (lagColumns.length !== 36 || windowColumns.length !== 24) {
  throw new Error(`Unexpected feature counts: lag=${lagColumns.length}, window=${windowColumns.length}`);
}

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("子任务6特征");
sheet.showGridLines = false;
sheet.getRangeByIndexes(0, 0, values.length, headers.length).values = values;
sheet.freezePanes.freezeRows(1);

const used = sheet.getUsedRange();
used.format.font = { name: "Arial", size: 10 };
used.format.verticalAlignment = "center";
sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};

for (const column of lagColumns) {
  sheet.getRangeByIndexes(0, column, values.length, 1).format.fill = "#FFF2CC";
  sheet.getCell(0, column).format.font = { name: "Arial", size: 10, bold: true, color: "#000000" };
}
for (const column of windowColumns) {
  sheet.getRangeByIndexes(0, column, values.length, 1).format.fill = "#D9EAF7";
  sheet.getCell(0, column).format.font = { name: "Arial", size: 10, bold: true, color: "#000000" };
}

sheet.getRangeByIndexes(1, 0, values.length - 1, 1).format.numberFormat = "yyyy-mm-dd";
for (let column = 1; column < headers.length; column += 1) {
  if (column !== 13) sheet.getRangeByIndexes(1, column, values.length - 1, 1).format.numberFormat = "0.0000";
}
sheet.getRangeByIndexes(0, 0, values.length, 1).format.columnWidth = 14;
sheet.getRangeByIndexes(0, 13, values.length, 1).format.columnWidth = 10;
for (let column = 1; column < headers.length; column += 1) {
  if (column !== 13) sheet.getRangeByIndexes(0, column, values.length, 1).format.columnWidth = 16;
}
sheet.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeight = 24;

workbook.recalculate();
const check = await workbook.inspect({
  kind: "table",
  range: "A1:P5",
  include: "values",
  tableMaxRows: 5,
  tableMaxCols: 16,
});
console.log(check.ndjson);

const preview = await workbook.render({
  sheetName: "子任务6特征",
  range: "A1:P10",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(os.tmpdir(), "acceptance1_feature_colors_preview.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, lagColumns: lagColumns.length, windowColumns: windowColumns.length }));
