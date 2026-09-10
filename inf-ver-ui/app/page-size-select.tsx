import { PAGE_SIZE_OPTIONS } from "./lib";

// Shared "rows per page" control used by every paginated table.
export default function PageSizeSelect({
  id,
  name,
  value
}: {
  id: string;
  name: string;
  value: number;
}) {
  return (
    <div className="filter-field">
      <label htmlFor={id}>Rows</label>
      <select id={id} name={name} defaultValue={value}>
        {PAGE_SIZE_OPTIONS.map((size) => (
          <option key={size} value={size}>
            {size}
          </option>
        ))}
      </select>
    </div>
  );
}
