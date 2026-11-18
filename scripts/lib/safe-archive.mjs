import { spawnSync } from "node:child_process";
import { lstat, readdir } from "node:fs/promises";
import path from "node:path";

function runListing(command, args, label) {
  const result = spawnSync(command, args, { encoding: "utf8" });
  if (result.error?.code === "ENOENT")
    throw new Error(`${label} requires ${command} to inspect archive member types`);
  if (result.status !== 0)
    throw new Error(result.stderr?.trim() || `${command} could not inspect ${label}`);
  return result.stdout;
}

function validateMemberPath(memberPath) {
  const normalized = memberPath.replaceAll("\\", "/");
  const segments = normalized.split("/");
  if (
    path.posix.isAbsolute(normalized) ||
    /^[a-z]:\//i.test(normalized) ||
    segments.includes("..")
  )
    throw new Error(`unsafe archive member path: ${memberPath}`);
}

function inspectZip(archive) {
  const members = runListing("unzip", ["-Z1", archive], "ZIP archive")
    .split(/\r?\n/)
    .filter(Boolean);
  if (members.length === 0) throw new Error(`archive contains no listable members: ${archive}`);
  for (const member of members) validateMemberPath(member);

  const metadata = runListing("zipinfo", ["-l", archive], "ZIP archive");
  const typedRows = metadata
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => /^[\-dlcbps][rwxStTs-]{9}\s/.test(line));
  if (typedRows.length !== members.length)
    throw new Error(
      `ZIP member types could not be verified for every entry: ${typedRows.length}/${members.length}`,
    );
  for (const row of typedRows) {
    const type = row[0];
    if (type !== "-" && type !== "d")
      throw new Error(`unsafe ZIP member type ${JSON.stringify(type)}: ${row}`);
  }
  return members;
}

function inspect7z(archive) {
  let listing;
  for (const command of ["7zz", "7z"]) {
    const result = spawnSync(command, ["l", "-ba", "-slt", archive], { encoding: "utf8" });
    if (result.error?.code === "ENOENT") continue;
    if (result.status !== 0)
      throw new Error(result.stderr?.trim() || `${command} could not inspect 7z archive`);
    listing = result.stdout;
    break;
  }
  if (listing === undefined)
    throw new Error("safe 7z extraction requires 7zz or 7z to inspect archive member types");

  const members = [];
  const blocks = listing.split(/\r?\n\r?\n/);
  for (const block of blocks) {
    const fields = new Map(
      block
        .split(/\r?\n/)
        .map((line) => line.match(/^([^=]+?) = (.*)$/))
        .filter(Boolean)
        .map((match) => [match[1].trim(), match[2]]),
    );
    const memberPath = fields.get("Path");
    if (!memberPath) continue;
    validateMemberPath(memberPath);
    if (fields.has("Symbolic Link") || fields.has("Hard Link"))
      throw new Error(`unsafe linked 7z member: ${memberPath}`);
    const mode = fields.get("Mode");
    if (mode && mode[0] !== "-" && mode[0] !== "d")
      throw new Error(`unsafe 7z member mode ${JSON.stringify(mode)}: ${memberPath}`);
    const attributes = fields.get("Attributes") ?? "";
    if (/[LRS]/.test(attributes))
      throw new Error(`unsafe 7z member attributes ${JSON.stringify(attributes)}: ${memberPath}`);
    if (!mode && !/[AD]/.test(attributes))
      throw new Error(`7z member type could not be verified: ${memberPath}`);
    members.push(memberPath);
  }
  if (members.length === 0) throw new Error(`archive contains no listable members: ${archive}`);
  return members;
}

export function assertSafeArchiveMembers(type, archive) {
  if (type === "zip") return inspectZip(archive);
  if (type === "7z") return inspect7z(archive);
  throw new Error(`unsupported archive type for safe extraction: ${type}`);
}

export async function validateExtractedTree(root) {
  const rootInfo = await lstat(root);
  if (!rootInfo.isDirectory()) throw new Error(`extraction root is not a directory: ${root}`);

  const visit = async (directory) => {
    for (const entry of await readdir(directory)) {
      const fullPath = path.join(directory, entry);
      const info = await lstat(fullPath);
      if (info.isSymbolicLink())
        throw new Error(`extracted symbolic link is not allowed: ${fullPath}`);
      if (info.isDirectory()) {
        await visit(fullPath);
        continue;
      }
      if (!info.isFile())
        throw new Error(`extracted special file is not allowed: ${fullPath}`);
      if (info.nlink !== 1)
        throw new Error(`extracted hard-linked file is not allowed: ${fullPath}`);
    }
  };

  await visit(root);
}
